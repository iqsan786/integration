import os
import cv2
import numpy as np
import logging
from datetime import datetime
from collections import deque
from ultralytics import YOLO

from shared.rabbitmq_utils import publish_to_queues

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.20
ORIGINAL_ROI_WIDTH = 766
ORIGINAL_ROI_HEIGHT = 350

WINDOW_SIZE = 40
YES_NEED = 0.25
NO_NEED = 0.88
MIN_DWELL = 30


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
# SYSTEM (MODULE ONLY)
# --------------------------------------------------
class SeatbeltSystem:
    SUPPORTED_ANALYTICS = {"ob-seat_belt_detection"}

    def __init__(self, config):

        self.model_paths = config["paths"]["model_paths"]

        self.model = None
        # self.model = YOLO(config["paths"]["model_paths"]["SEATBELT"])
        self.current_object = None

        self.EVENT_DIR = config["paths"]["event_dir"]
        os.makedirs(self.EVENT_DIR, exist_ok=True)

        # temporal logic
        self.window = deque(maxlen=WINDOW_SIZE)
        self.state = "NO"
        self.frames_in_state = 0    

        logger.info("[INIT] Seatbelt System Ready")


    def scale_roi(self, roi, w, h):
        return [
            ((x / ORIGINAL_ROI_WIDTH) * w,
             (y / ORIGINAL_ROI_HEIGHT) * h)
            for x, y in roi
        ]


    def save_frame(self, frame, date_obj, ip, frame_id):

        path = os.path.join(
            self.EVENT_DIR,
            ip.replace(".", "_"),
            date_obj.strftime("%Y-%m-%d"),
            f"{date_obj.hour:02d}-{(date_obj.hour+1)%24:02d}"
        )

        os.makedirs(path, exist_ok=True)

        filename = f"seatbelt_{date_obj.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{frame_id}.jpg"

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


    def detect(self, frame):

        best_conf = 0.0
        best_box = None

        results = self.model(frame, conf=MIN_CONFIDENCE, verbose=False)

        if results and results[0].boxes is not None:

            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()

            idx = int(confs.argmax())
            best_conf = float(confs[idx])

            b = boxes.xyxy[idx].cpu().numpy()
            best_box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        return best_conf, best_box


    def update_state(self, conf):

        self.window.append(conf)

        w = list(self.window)
        n = len(w)

        yes_count = sum(1 for c in w if c >= MIN_CONFIDENCE)
        yes_frac = yes_count / n if n > 0 else 0

        self.frames_in_state += 1

        if self.frames_in_state >= MIN_DWELL:

            if self.state == "NO" and yes_frac >= YES_NEED:
                self.state = "YES"
                self.frames_in_state = 0

            elif self.state == "YES" and (1 - yes_frac) >= NO_NEED:
                self.state = "NO"
                self.frames_in_state = 0

        return self.state


    # 🔥 MAIN ENTRY (called from main.py)
    def process_frame(self, frame, msg):

        camera_config = msg.get("camera_config", {})

        reader_id = msg["reader_id"]
        frame_id = msg["frame_id"]
        timestamp = msg["timestamp"]
        camera_ip = msg["ip_address"]

        zone_id = msg.get("zone_id")
        location_id = msg.get("location_id")

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

                    analytics_list.append({
                        "type": st.get("name"),
                        "object": values[0].lower(),
                        "attributeName": attr_maps[0].get("attributeName"),
                        "attributeValue": values
                    })

        if not analytics_list:
            return frame


        # ---------------- LOAD MODEL ----------------
        object_name = "seatbelt"

        model_path = self.model_paths.get(object_name)

        if not model_path:
            logger.warning(f"No model for {object_name}")
            return frame

        if self.current_object != object_name or self.model is None:
            self.model = YOLO(model_path)
            self.current_object = object_name
            logger.info(f"[MODEL] Loaded {object_name}")


        # ---------------- ROI ----------------
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
        conf, box = self.detect(frame)


        # ---------------- ANALYTICS ----------------
        for analytics in analytics_list:

            atype = analytics["type"]
            roi_polygon = roi_map.get(atype)

            if box and roi_polygon:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                if not point_in_polygon(cx, cy, roi_polygon):
                    continue

            state = self.update_state(conf)
            success = True if state == "YES" else False

            if success:
                continue

            date_obj = datetime.fromtimestamp(timestamp)

            frame_path = self.save_frame(frame, date_obj, camera_ip, frame_id)
            crop_paths = []
            for i, det in enumerate(box):
                crop = self.save_crop(frame, det["bbox"], date_obj, camera_ip, i)
                if crop:
                    crop_paths.append(crop)


            payload = {
                "readerId": reader_id,
                "type": "object_analytics",
                "attributeName": analytics["attributeName"],
                "attributeValue": analytics["attributeValue"],
                "frameId": frame_id,
                "zoneId": zone_id,
                "locationId": location_id,
                "detectionTime": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "frameLocation": frame_path,
                "roiCoordinates": [{"x": x, "y": y} for x, y in roi_polygon] if roi_polygon else None,
                "success": success,
                "crops": crop_paths
            }

            publish_to_queues(payload)

            logger.info("[PUBLISH] Seatbelt violation")

        return frame
    