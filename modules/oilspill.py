import os
import cv2
import numpy as np
import logging
from datetime import datetime
from ultralytics import YOLO

from shared.rabbitmq_utils import publish_to_queues

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.35
ORIGINAL_ROI_WIDTH = 766
ORIGINAL_ROI_HEIGHT = 350


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
                        xinters = (y - py1)*(px2 - px1)/(py2 - py1) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside

        px1, py1 = px2, py2

    return inside

class OilSpillSystem:

    def __init__(self, config):

        self.model_paths = config["paths"]["model_paths"]

        self.model = None
        # self.model = YOLO(config["paths"]["model_paths"]["OILSPILL"])
        self.current_object = None

        self.EVENT_DIR = config["paths"]["event_dir"]
        os.makedirs(self.EVENT_DIR, exist_ok=True)

        logger.info("[INIT] Oil Spill System Ready")


    def scale_roi(self, roi, w, h):
        return [((x / ORIGINAL_ROI_WIDTH) * w,
                 (y / ORIGINAL_ROI_HEIGHT) * h) for x, y in roi]


    def save_frame(self, frame, date_obj, ip, frame_id):

        path = os.path.join(
            self.EVENT_DIR,
            ip.replace(".", "_"),
            date_obj.strftime("%Y-%m-%d"),
            f"{date_obj.hour:02d}-{(date_obj.hour+1)%24:02d}"
        )

        os.makedirs(path, exist_ok=True)

        filename = f"oilspill_{date_obj.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{frame_id}.jpg"
        full_path = os.path.join(path, filename)

        cv2.imwrite(full_path, frame)
        return os.path.abspath(full_path)


    def detect(self, frame):

        results = self.model(frame, conf=MIN_CONFIDENCE, verbose=False)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

        return detections


    def process_frame(self, frame, msg):

        camera_config = msg.get("camera_config", {})

        reader_id = msg["reader_id"]
        frame_id = msg["frame_id"]
        timestamp = msg["timestamp"]
        camera_ip = msg["ip_address"]

        zone_id = msg.get("zone_id")
        location_id = msg.get("location_id")

        h, w = frame.shape[:2]

        # ---------------- ANALYTICS EXTRACTION ----------------
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

                    analytics_list.append({
                        "type": st.get("name"),
                        "object": values[0].lower(),   # 🔥 IMPORTANT
                        "attributeName": attr_maps[0].get("attributeName"),
                        "attributeValue": values
                    })

        if not analytics_list:
            return frame


        # ---------------- MODEL LOAD (DYNAMIC ✅) ----------------
        object_name = analytics_list[0]["object"]

        if object_name not in self.model_paths:
            logger.warning(f"No model for {object_name}")
            return frame

        if self.current_object != object_name or self.model is None:
            self.model = YOLO(self.model_paths[object_name])
            self.current_object = object_name
            logger.info(f"[MODEL] Loaded {object_name}")


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


        # ---------------- DETECTION ----------------
        detections = self.detect(frame)


        # ---------------- PROCESS ANALYTICS ----------------
        for analytics in analytics_list:

            atype = analytics["type"]
            roi_polygon = roi_map.get(atype)

            valid = []

            if roi_polygon:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    if point_in_polygon(cx, cy, roi_polygon):
                        valid.append(det)
            else:
                valid = detections

            success = len(valid) > 0

            if not success:
                continue


            # ---------------- SAVE + PUBLISH ----------------
            date_obj = datetime.fromtimestamp(timestamp)

            frame_path = self.save_frame(frame, date_obj, camera_ip, frame_id)

            payload = {
                "readerId": reader_id,
                "type": atype,   # 🔥 consistent
                "attributeName": analytics["attributeName"],
                "attributeValue": analytics["attributeValue"],
                "frameId": frame_id,
                "zoneId": zone_id,
                "locationId": location_id,
                "detectionTime": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "frameLocation": frame_path,
                "roiCoordinates": [{"x": x, "y": y} for x, y in roi_polygon] if roi_polygon else None,
                "detections": valid,
                "success": success
            }

            publish_to_queues(payload)

            logger.info("[PUBLISH] Oil Spill Detected")

        return frame