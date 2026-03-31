import os
import cv2
import numpy as np
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

from shared.rabbitmq_utils import publish_to_queues

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.12
ORIGINAL_ROI_WIDTH = 766
ORIGINAL_ROI_HEIGHT = 350


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
# CLASSIFIER
# --------------------------------------------------
class DistractedDriverClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.mobilenet_v2(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        self.model = model

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# SYSTEM (MODULE)
# --------------------------------------------------
class PhoneDetectionSystem:

    def __init__(self, config):

        self.EVENT_DIR = config["paths"]["event_dir"]
        self.model_paths = config["paths"]["model_paths"]

        os.makedirs(self.EVENT_DIR, exist_ok=True)

        self.model = None
        # self.model = YOLO(config["paths"]["model_paths"]["PHONE"])
        self.current_object = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 🔥 CLASSIFIER LOAD (once)
        self.classifier = DistractedDriverClassifier()
        self.classifier.load_state_dict(
            torch.load(config["paths"]["classifier_model"], map_location=self.device)
        )
        self.classifier.to(self.device)
        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("[INIT] Phone Detection Ready")


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

        filename = f"phone_detection_{date_obj.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{frame_id}.jpg"

        full_path = os.path.join(path, filename)

        cv2.imwrite(full_path, frame)

        return os.path.abspath(full_path)


    # --------------------------------------------------
    def detect(self, frame):

        results = self.model(frame, conf=MIN_CONFIDENCE, verbose=False)[0]

        drivers = []
        phones = []

        if results.boxes is None:
            return drivers, phones

        for box in results.boxes:

            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if cls == 0:  # driver
                drivers.append((x1, y1, x2, y2, conf))

            elif cls == 67:  # phone
                phones.append((x1, y1, x2, y2, conf))

        return drivers, phones


    def classify_driver(self, crop):

        img = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(img)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        return pred, float(probs[pred])


    # 🔥 MAIN ENTRY
    def process_frame(self, frame, msg):

        camera_config = msg.get("camera_config", {})

        reader_id = msg["reader_id"]
        frame_id = msg["frame_id"]
        timestamp = msg["timestamp"]
        camera_ip = msg["ip_address"]

        zone_id = msg.get("zone_id")
        location_id = msg.get("location_id")

        h, w = frame.shape[:2]

        # ---------------- ANALYTICS ----------------
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


        # ---------------- MODEL LOAD ----------------
        object_name = analytics_list[0]["object"]

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
        drivers, phones = self.detect(frame)

        if not drivers:
            return frame

        driver = max(drivers, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        x1, y1, x2, y2, _ = driver

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame

        pred, conf = self.classify_driver(crop)
        distracted = (pred == 1)

        # phone overlap
        for px1, py1, px2, py2, _ in phones:
            if not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2):
                distracted = True
                break

        if not distracted:
            return frame


        # ---------------- PUBLISH ----------------
        for analytics in analytics_list:

            atype = analytics["type"]
            roi_polygon = roi_map.get(atype)

            if roi_polygon:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if not point_in_polygon(cx, cy, roi_polygon):
                    continue

            date_obj = datetime.fromtimestamp(timestamp)

            frame_path = self.save_frame(frame, date_obj, camera_ip, frame_id)

            payload = {
                "readerId": reader_id,
                "type": "phone_detection",
                "attributeName": analytics["attributeName"],
                "attributeValue": analytics["attributeValue"],
                "frameId": frame_id,
                "zoneId": zone_id,
                "locationId": location_id,
                "detectionTime": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "frameLocation": frame_path,
                "roiCoordinates": [{"x": x, "y": y} for x, y in roi_polygon] if roi_polygon else None,
                "success": distracted,
                "confidence": conf
            }

            publish_to_queues(payload)

            logger.info("[PUBLISH] Phone Usage Detected")

        return frame