import os
import json
import cv2
import numpy as np
import pika
import pickle
import argparse
import traceback
import logging
from datetime import datetime
from ultralytics import YOLO

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.25
ORIGINAL_ROI_WIDTH = 766
ORIGINAL_ROI_HEIGHT = 350

SMOOTH_FRAMES = {
    "POTHOLE": 3,
    "POLE": 3,
    "BRANCH": 3,
    "WATERLOG": 6
}

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.rabbitmq_utils import publish_to_queues


# --------------------------------------------------
# ROI CHECK
# --------------------------------------------------
def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    px1, py1 = polygon[0]

    for i in range(n + 1):
        px2, py2 = polygon[i % n]

        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside

        px1, py1 = px2, py2

    return inside


# --------------------------------------------------
# SYSTEM
# --------------------------------------------------
class ObstacleDetectionSystem:
    SUPPORTED_ANALYTICS = {"ob-anomoly_detection"}

    def __init__(self, config):

        self.EVENT_DIR = config["paths"]["event_dir"]

        self.model = YOLO(config["paths"]["model_paths"]["obstacle"])

        os.makedirs(self.EVENT_DIR, exist_ok=True)

        # temporal smoothing
        self.counts = {k: 0 for k in SMOOTH_FRAMES}

        logger.info("[INIT] Obstacle Detection Ready")

    # --------------------------------------------------
    def scale_roi(self, roi, w, h):
        return [((x / ORIGINAL_ROI_WIDTH) * w,
                 (y / ORIGINAL_ROI_HEIGHT) * h) for x, y in roi]

    # --------------------------------------------------
    def save_frame(self, frame, date_obj, ip, frame_id):

        path = os.path.join(
            self.EVENT_DIR,
            ip.replace(".", "_"),
            date_obj.strftime("%Y-%m-%d"),
            f"{date_obj.hour:02d}-{(date_obj.hour+1)%24:02d}"
        )

        os.makedirs(path, exist_ok=True)

        filename = f"obstacle_{date_obj.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{frame_id}.jpg"
        full_path = os.path.join(path, filename)

        cv2.imwrite(full_path, frame)

        return os.path.abspath(full_path)
    
    def save_crop(self, frame, bbox, date_obj, ip):

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        path = os.path.join(
            self.crop_dir,
            ip.replace(".", "_"),
            date_obj.strftime("%Y-%m-%d"),
            f"{date_obj.hour:02d}-{(date_obj.hour+1)%24:02d}"
        )

        os.makedirs(path, exist_ok=True)

        filename = f"{self.current_object}_{date_obj.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        full_path = os.path.join(path, filename)

        cv2.imwrite(full_path, crop)
        return os.path.abspath(full_path)

    # --------------------------------------------------
    def detect_objects(self, frame, object_types):

        results = self.model(frame, imgsz=960, conf=MIN_CONFIDENCE, iou=0.6)[0]

        detections = []

        if results.boxes is None:
            return detections

        h, w = frame.shape[:2]
        frame_area = h * w

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = self.model.names[cls].replace(" ", "").upper()

            if label not in object_types:
                continue

            box_area = (x2 - x1) * (y2 - y1)

            # global filter
            if box_area > 0.35 * frame_area:
                continue

            # perspective filter
            y_center = (y1 + y2) / 2
            depth_ratio = y_center / h
            max_allowed_area = frame_area * (0.25 * (1 - depth_ratio) + 0.08)

            if box_area > max_allowed_area:
                continue

            # class-specific filters
            if label == "POTHOLE":
                if conf < 0.60 or y2 < h * 0.55 or box_area < 1800:
                    continue

            elif label == "WATERLOG":
                if conf < 0.50 or y2 < h * 0.60 or box_area < 3000:
                    continue

            elif label == "POLE":
                box_h = y2 - y1
                box_w = x2 - x1
                if conf < 0.50 or (box_h / (box_w + 1)) < 2.0 or y2 > h * 0.93:
                    continue

            elif label == "BRANCH":
                if conf < 0.30:
                    continue

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections

    # --------------------------------------------------
    def process_frame(self, frame, msg):

        reader_id = msg["reader_id"]
        frame_id = msg["frame_id"]
        timestamp = msg["timestamp"]
        camera_ip = msg["ip_address"]

        zone_id = msg.get("zone_id")
        location_id = msg.get("location_id")

        camera_config = msg.get("camera_config", {})

        h, w = frame.shape[:2]

        # ---------------- EXTRACT ANALYTICS ----------------
        analytics_list = []

        for group in camera_config.get("readerMetrics", []):
            if group.get("metricType") != "object_analytics":
                continue

            for metric in group.get("metricJson", []):
                for st in metric.get("subType", []):

                    if not st.get("active"):
                        continue

                    attr_maps = st.get("attributeMaps", [])
                    if not attr_maps:
                        continue

                    values = attr_maps[0].get("attributeValue", [])
                    if not values:
                        continue

                    object_types = [v.replace(" ", "").upper() for v in values]

                    
                    if not any(obj in SMOOTH_FRAMES for obj in object_types):
                        continue

                    analytics_list.append({
                        "type": st.get("name"),
                        "objects": object_types,
                        "attributeName": attr_maps[0].get("attributeName"),
                        "attributeValue": values
                    })

        if not analytics_list:
            return frame

        # ---------------- ROI MAP ----------------
        roi_map = {}

        for roi in camera_config.get("readerRois", []):

            roi_json = roi.get("roiJson", {})

            if not roi_json.get("active"):
                continue

            coords = roi_json.get("coordinates", [])
            if not coords:
                continue

            polygon = [
                (float(c["xcoordinate"]), float(c["ycoordinate"]))
                for c in sorted(coords, key=lambda x: x["orderPosition"])
            ]

            scaled = self.scale_roi(polygon, w, h)

            for a in roi_json.get("analytics", []):
                roi_map[a] = scaled

        # ---------------- PROCESS ----------------
        for analytics in analytics_list:

            atype = analytics["type"]
            object_types = analytics["objects"]

            roi_polygon = roi_map.get(atype)

            detections = self.detect_objects(frame, object_types)

            # ROI filtering
            if roi_polygon:
                valid = []
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    if point_in_polygon(cx, cy, roi_polygon):
                        valid.append(det)
            else:
                valid = detections

            if not valid:
                continue

            # temporal smoothing
            detected_now = {k: False for k in self.counts}

            for d in valid:
                if d["label"] in detected_now:
                    detected_now[d["label"]] = True

            for k in self.counts:
                self.counts[k] = self.counts[k] + 1 if detected_now[k] else 0

            warnings = [
                k for k in self.counts
                if self.counts[k] >= SMOOTH_FRAMES[k]
            ]

            if not warnings:
                continue

            # SAVE + PUBLISH
            date_obj = datetime.fromtimestamp(timestamp)
            frame_path = self.save_frame(frame, date_obj, camera_ip, frame_id)
            
            crop_paths = []
            for i, det in enumerate(valid):
                crop = self.save_crop(frame, det["bbox"], date_obj, camera_ip, i)
                if crop:
                    crop_paths.append(crop)


            payload = {
                "readerId": reader_id,
                "type": "object_detection",
                "attributeName": analytics["attributeName"],
                "attributeValue": analytics["attributeValue"],
                "frameId": frame_id,
                "zoneId": zone_id,
                "locationId": location_id,
                "detectionTime": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "frameLocation": frame_path,
                "roiCoordinates": [{"x": x, "y": y} for x, y in roi_polygon] if roi_polygon else None,
                "detections": valid,
                "success": True,
                "crops": crop_paths
            }

            publish_to_queues(payload)

            logger.info(f"[PUBLISH] Obstacles: {warnings}")

        return frame


