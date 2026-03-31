import os
import cv2
import numpy as np
import logging
from datetime import datetime
from ultralytics import YOLO
import redis

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

class TyreSystem:

    def __init__(self, config):

        self.model_paths = config["paths"]["model_paths"]
        # self.model = YOLO(config["paths"]["model_paths"]["TYRE"])
        self.model = None
        self.current_object = None

        self.EVENT_DIR = config["paths"]["event_dir"]
        os.makedirs(self.EVENT_DIR, exist_ok=True)

        self.redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self.REDIS_TTL = 60

        self.previous_y = {}
        self.total_count = 0

        logger.info("[INIT] Tyre System Ready")


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

        filename = f"tyre_{date_obj.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{frame_id}.jpg"
        full_path = os.path.join(path, filename)

        cv2.imwrite(full_path, frame)
        return os.path.abspath(full_path)


    def detect(self, frame, reader_id):

        results = self.model.track(
            frame,
            conf=MIN_CONFIDENCE,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:

            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id[0])

            detections.append({
                "track_id": f"{reader_id}_{tid}",
                "bbox": [x1, y1, x2, y2]
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

        # ---------------- EXTRACT ANALYTICS ----------------
        analytics_list = []

        for group in camera_config.get("readerMetrics", []):
            if group.get("metricType") != "object_analytics":
                continue

            for metric in group.get("metricJson", []):
                for st in metric.get("subType", []):

                    if not st.get("active"):
                        continue

                    values = st.get("attributeMaps", [{}])[0].get("attributeValue", [])

                    if values and values[0].lower() == "tyre":
                        analytics_list.append({
                            "type": st.get("name"),
                            "object": "tyre"
                        })

        if not analytics_list:
            return frame


        # ---------------- LOAD MODEL ----------------
        object_name=analytics_list[0]["object"]
        model_path=self.model_paths.get(object_name)
        if not model_path:
            logger.warning(f"No model for {object_name}")
            return frame
        if self.current_object != object_name or self.model is None:
            self.model = YOLO(model_path)
            self.current_object = object_name
            logger.info(f"[MODEL] Loaded {object_name}")
        


        # ---------------- ROI + LINE ----------------
        roi_polygon = None
        line_y = None

        for roi in camera_config.get("readerRois", []):

            roi_json = roi.get("roiJson", {})

            if not roi_json.get("active"):
                continue


            coords = roi_json.get("coordinates", [])
            polygon = [(float(c["xcoordinate"]), float(c["ycoordinate"])) for c in coords]
            roi_polygon = self.scale_roi(polygon, w, h)

            if roi_json.get("type") == "direction":
                ref = roi_json.get("referPoint")
                if ref and "yCoordinated:" in ref:
                    raw_y = int(ref.split("yCoordinated:")[1].strip())
                    line_y = (raw_y / ORIGINAL_ROI_HEIGHT) * h


        detections = self.detect(frame, reader_id)

        valid = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if roi_polygon and not point_in_polygon(cx, cy, roi_polygon):
                continue

            valid.append(det)

        if not valid:
            return frame


        # ---------------- COUNTING LOGIC ----------------
        success = False

        for det in valid:

            track_id = det["track_id"]
            cy = (det["bbox"][1] + det["bbox"][3]) / 2

            if track_id not in self.previous_y:
                self.previous_y[track_id] = cy
                continue

            redis_key = f"counted:{track_id}"

            if self.previous_y[track_id] > line_y and cy <= line_y:
                if not self.redis.exists(redis_key):
                    self.total_count += 1
                    self.redis.setex(redis_key, self.REDIS_TTL, 1)
                    success = True

            self.previous_y[track_id] = cy


        if not success:
            return frame


        # ---------------- SAVE ----------------
        date_obj = datetime.fromtimestamp(timestamp)

        frame_path = self.save_frame(frame, date_obj, camera_ip, frame_id)

        payload = {
            "readerId": reader_id,
            "type": "object_analytics",
            "frameId": frame_id,
            "zoneId": zone_id,
            "locationId": location_id,
            "frameLocation": frame_path,
            "detectionTime": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
            "success": success,
            "TotalCount": self.total_count
        }

        publish_to_queues(payload)

        logger.info(f"[PUBLISH] Tyre Count = {self.total_count}")

        return frame