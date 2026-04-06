import os
import sys
import json
import cv2
import numpy as np
import pika
import pickle
import redis
import argparse
import traceback
import logging
import torch
import math
from datetime import datetime
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import ultralytics.utils.loss

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.25
ORIGINAL_ROI_WIDTH = 766
ORIGINAL_ROI_HEIGHT = 350

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from shared.rabbitmq_utils import publish_to_queues
except ImportError:
    logger.warning("Could not import publish_to_queues. Using mock publisher.")


    def publish_to_queues(payload):
        logger.info(f"[MOCK PUBLISH] {json.dumps(payload)}")

# =====================================================
# HOTFIX: PATCH 'DFLoss' ERROR
# =====================================================
if not hasattr(ultralytics.utils.loss, 'DFLoss'):
    class DFLoss:
        def __init__(self, reg_max=16): pass

        def __call__(self, pred, target): pass


    ultralytics.utils.loss.DFLoss = DFLoss


# --------------------------------------------------
# Geometry & Polygon ROI Check
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


def iou(bb_test, bb_gt):
    xx1, yy1 = max(bb_test[0], bb_gt[0]), max(bb_test[1], bb_gt[1])
    xx2, yy2 = min(bb_test[2], bb_gt[2]), min(bb_test[3], bb_gt[3])
    w, h = max(0., xx2 - xx1), max(0., yy2 - yy1)
    wh = w * h
    return wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                 (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)


def compute_machine_representative_point(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


# =====================================================
# TRACKING CLASSES (Kalman & Sort)
# =====================================================
class KalmanMotion:
    def __init__(self, dt=1 / 30.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.P *= 10.
        self.kf.R *= 10.
        self.kf.Q *= 0.1
        self.history = deque(maxlen=30)
        self.smooth_vx, self.smooth_vy = 0.0, 0.0

    def update(self, cx, cy):
        self.kf.update([cx, cy])
        self.history.append((cx, cy))

    def predict_future(self, steps=90):
        if len(self.history) < 2: return []
        curr_x, curr_y = self.history[-1]
        frames_back = min(len(self.history) - 1, 15)
        old_x, old_y = self.history[-frames_back - 1]

        raw_vx = (curr_x - old_x) / float(frames_back)
        raw_vy = (curr_y - old_y) / float(frames_back)

        self.smooth_vx = (0.1 * raw_vx) + (0.9 * self.smooth_vx)
        self.smooth_vy = (0.1 * raw_vy) + (0.9 * self.smooth_vy)

        MAX_SPEED = 3.5
        vx = max(-MAX_SPEED, min(MAX_SPEED, self.smooth_vx))
        vy = max(-MAX_SPEED, min(MAX_SPEED, self.smooth_vy))

        return [(int(curr_x + (vx * i)), int(curr_y + (vy * i))) for i in range(1, steps + 1)]


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.P *= 10.;
        self.kf.R *= 1.;
        self.kf.Q *= 0.01
        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.time_since_update, self.hits, self.hit_streak, self.age = 0, 0, 0, 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array(bbox).reshape((4, 1)))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x


def associate_detections_to_trackers(detections, trackers, iou_threshold):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers): iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.atleast_2d(np.array(list(zip(*matched_indices))))
    if matched_indices.shape[1] == 0: matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0]);
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    return (np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)), np.array(
        unmatched_dets), np.array(unmatched_trks)


class SortTracker:
    def __init__(self, max_age=25, min_hits=3, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.trackers, self.frame_count = [], 0

    def update(self, detections):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 4))
        to_del, ret = [], []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos[:4].reshape(-1)
            if np.any(np.isnan(trk)): to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del): self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trks, self.iou_threshold)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                trk.update(detections[matched[np.where(matched[:, 1] == t)[0], 0][0]])
        for i in unmatched_dets: self.trackers.append(KalmanBoxTracker(detections[i]))
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.kf.x[:4].reshape(-1)
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append([*d, trk.id])
            i -= 1
            if trk.time_since_update > self.max_age: self.trackers.pop(i)
        return np.array(ret)


# --------------------------------------------------
# Detection System
# --------------------------------------------------
class TrajectoryDetectionSystem:
    SUPPORTED_ANALYTICS = {"ob-trajectory_detection"}

    def __init__(self, config):
        self.model_paths = config["paths"]["model_paths"]
        self.redis = redis.Redis(host="localhost", port=6379, db=0)

        self.model_machines = None
        self.model_humans = None
        self.device = '0' if torch.cuda.is_available() else 'cpu'

        # Construction Tracking State
        self.tracker = SortTracker()
        self.body_kalman = {}
        self.machine_memory = {}
        self.machine_seen_count = defaultdict(int)

        # in_out tracking
        self.previous_y = {}

        # occupancy tracking
        self.current_count = 0
        self.max_value = None
        self.last_publish_time = 0
        self.last_alert_time = 0

        logger.info("[INIT] System Ready")
    
    def save_frame(self, frame, date_obj, ip, frame_id):

        path = os.path.join(
            self.EVENT_DIR,
            ip.replace(".", "_"),
            date_obj.strftime("%Y-%m-%d"),
            f"{date_obj.hour:02d}-{(date_obj.hour+1)%24:02d}"
        )

        os.makedirs(path, exist_ok=True)

        filename = f"suspended_load_{date_obj.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{frame_id}.jpg"

        full_path = os.path.join(path, filename)

        cv2.imwrite(full_path, frame)
        return os.path.abspath(full_path)
    
        
    def save_crop(self, frame, bbox, date_obj, ip, label):

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        path = os.path.join("crops", label)
        os.makedirs(path, exist_ok=True)

        filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        full_path = os.path.join(path, filename)

        cv2.imwrite(full_path, crop)
        return full_path

    # --------------------------------------------------
    def scale_roi(self, roi, w, h):
        return [((x / ORIGINAL_ROI_WIDTH) * w, (y / ORIGINAL_ROI_HEIGHT) * h) for x, y in roi]

    # --------------------------------------------------
    def process_frame(self, frame, msg):
        reader_id = msg["reader_id"]
        timestamp = msg["timestamp"]
        frame_id = msg["frame_id"]
        zone_id = msg.get("zone_id")
        timestamp = msg["timestamp"]
        location_id = msg.get("location_id")
        camera_config = msg.get("camera_config", {})
        h, w = frame.shape[:2]

        # ---------------- EXTRACT ANALYTICS ----------------
        analytics_list = []
        for metric_group in camera_config.get("readerMetrics", []):
            if metric_group.get("metricType") != "object_analytics":
                continue
            for metric in metric_group.get("metricJson", []):
                
                    for st in metric.get("subType", []):
                        if not st.get("active"): continue
                        attr_maps = st.get("attributeMaps", [])
                        if not attr_maps: continue
                        values = attr_maps[0].get("attributeValue", [])
                        if not values: continue

                        analytics_list.append({
                            "type": st["name"],
                            "object": values[0].lower(),
                            "maxValue": st.get("maxValue")
                        })

        # ---------------- LOAD MODEL ----------------
        # Load Dual Construction Models if not loaded
        if self.model_machines is None:
            machine_path = self.model_paths.get("machine_model", "dbconst.pt")
            human_path = self.model_paths.get("human_model", "yolov8x.pt")

            logger.info(f"[MODEL] Loading {machine_path} and {human_path}")
            self.model_machines = YOLO(machine_path)
            self.model_humans = YOLO(human_path)
            if self.device == '0':
                self.model_machines.to('cuda')
                self.model_humans.to('cuda')

        # ---------------- ROI MAP ----------------
        roi_map = {}
        for roi in camera_config.get("readerRois", []):
            roi_json = roi.get("roiJson", {})
            if not roi_json.get("active"): continue
            coords = roi_json.get("coordinates", [])
            if not coords: continue

            polygon = [(float(c["xcoordinate"]), float(c["ycoordinate"]))
                       for c in sorted(coords, key=lambda x: x["orderPosition"])]
            scaled = self.scale_roi(polygon, w, h)

            for a in roi_json.get("analytics", []):
                roi_map[a] = scaled

        # ---------------- DETECTION & TRACKING ----------------
        # 1. Machines
        results_machines = self.model_machines(frame, conf=0.25, device=self.device, verbose=False)
        detections = []
        machine_boxes = []

        for r in results_machines:
            if r.boxes is None: continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                label = self.model_machines.names[int(b.cls[0])] if hasattr(self.model_machines, 'names') else "machine"
                if ((x2 - x1) * (y2 - y1)) < 1000: continue
                machine_boxes.append([x1, y1, x2, y2])
                detections.append((x1, y1, x2, y2, label, conf))

        # 2. Humans
        results_humans = self.model_humans(frame, classes=[0], conf=0.35, device=self.device, verbose=False)
        raw_humans = []
        for r in results_humans:
            if r.boxes is None: continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                raw_humans.append((x1, y1, x2, y2, float(b.conf[0])))

        # 3. Track Machines
        tracked = self.tracker.update(machine_boxes)
        active_ids = set()

        for (x1, y1, x2, y2, tid) in tracked:
            active_ids.add(tid)
            self.machine_seen_count[tid] += 1
            if self.machine_seen_count[tid] < 3: continue

            # Match label to tracker
            current_label = "machine"
            t_cx, t_cy = (x1 + x2) / 2, (y1 + y2) / 2
            best_dist = 10000
            for (dx1, dy1, dx2, dy2, d_label, d_conf) in detections:
                dist = abs(t_cx - (dx1 + dx2) / 2) + abs(t_cy - (dy1 + dy2) / 2)
                if dist < best_dist and dist < 100:
                    best_dist, current_label = dist, d_label

            self.machine_memory[tid] = {"bbox": [x1, y1, x2, y2], "label": current_label, "life": 30}

        # Cleanup memory
        for tid in list(self.machine_memory.keys()):
            if tid not in active_ids:
                self.machine_memory[tid]["life"] -= 1
                if self.machine_memory[tid]["life"] <= 0:
                    self.machine_memory.pop(tid);
                    self.body_kalman.pop(tid, None)

        # ---------------- PROCESS ANALYTICS ----------------
        valid_humans = []
        for (x1, y1, x2, y2, conf) in raw_humans:
            hx, hy = (x1 + x2) // 2, y2
            valid_humans.append(
                {"track_id": f"{reader_id}_{x1}_{y1}", "bbox": [x1, y1, x2, y2], "point": (hx, hy), "conf": conf})

        for analytics in analytics_list:
            atype = analytics["type"]
            obj = analytics["object"]
            roi_polygon = roi_map.get(atype)

            # Filter valid objects inside ROI (if ROI exists for this analytic)
            roi_filtered_objects = []
            if roi_polygon:
                for det in valid_humans:
                    cx, cy = det["point"]
                    if point_in_polygon(cx, cy, roi_polygon):
                        roi_filtered_objects.append(det)
            else:
                roi_filtered_objects = valid_humans  # Default to all if no ROI provided

            # ==================================================
            # OCCUPANCY COUNT
            # ==================================================
            if atype == "ob-object_occupancy_count":
                self.current_count = len(roi_filtered_objects)
                self.max_value = analytics["maxValue"]

                if self.max_value and self.current_count > self.max_value:
                    if timestamp - self.last_alert_time > 30:
                        self.last_alert_time = timestamp
                        self.publish_event(reader_id, frame_id, zone_id, location_id, roi_polygon, obj, False,
                                           self.current_count)

                if timestamp - self.last_publish_time > 30:
                    self.last_publish_time = timestamp
                    self.publish_event(reader_id, frame_id, zone_id, location_id, roi_polygon, obj, True,
                                       self.current_count)

            # ==================================================
            # IN-OUT LOGIC
            # ==================================================
            elif atype == "ob-object_in_out" and roi_polygon:
                line_y = roi_polygon[0][1]
                for det in roi_filtered_objects:
                    tid, cy = det["track_id"], det["point"][1]
                    if tid not in self.previous_y:
                        self.previous_y[tid] = cy
                        continue
                    if self.previous_y[tid] > line_y and cy <= line_y:
                        self.publish_event(reader_id, frame_id, zone_id, location_id, roi_polygon, obj, True, 1)
                    self.previous_y[tid] = cy

            # ==================================================
            # CONSTRUCTION SAFETY LOGIC (Worker vs Machine)
            # ==================================================
            elif atype == "construction_safety":
                alerts = []
                for tid, data in self.machine_memory.items():
                    x1, y1, x2, y2 = map(int, data["bbox"])
                    body_point = compute_machine_representative_point([x1, y1, x2, y2])

                    if tid not in self.body_kalman: self.body_kalman[tid] = KalmanMotion()
                    self.body_kalman[tid].update(body_point[0], body_point[1])
                    future_points = self.body_kalman[tid].predict_future(steps=90)

                    # Compute Dynamic ROI based on machine type
                    rx, ry = 60, 40  # Default
                    if "excavator" in data["label"]:
                        rx, ry = 120, 80
                    elif "truck" in data["label"]:
                        rx, ry = 0, 0  # Trucks use path prediction

                    for human in valid_humans:
                        hx, hy = human["point"]

                        # 1. Dynamic Ellipse Danger Check
                        in_danger = False
                        if rx > 0:
                            term_x, term_y = ((hx - body_point[0]) ** 2) / (rx ** 2), ((hy - body_point[1]) ** 2) / (
                                        ry ** 2)
                            if (term_x + term_y) <= 1.0: in_danger = True

                        # 2. Predicted Path Danger Check
                        in_path = False
                        path_radius = max((x2 - x1) * 0.8, 120)
                        for px, py in future_points:
                            if math.sqrt((hx - px) ** 2 + (hy - py) ** 2) < path_radius: in_path = True

                        if in_danger or in_path:
                            alerts.append({"machine_id": tid, "violation": "Collision Risk"})
                            
                            date_obj = datetime.fromtimestamp(timestamp)
                            camera_ip = msg.get("ip_address", "unknown")

                            # 🔹 crop human
                            self.save_crop(frame, human["bbox"], date_obj, camera_ip, "human")

                            # 🔹 crop machine
                            self.save_crop(frame, data["bbox"], date_obj, camera_ip, "machine")

                if alerts and timestamp - self.last_alert_time > 5:
                    self.last_alert_time = timestamp
                    self.publish_event(reader_id, frame_id, zone_id, location_id, roi_polygon, "SafetyViolation", False,
                                       len(alerts))

        return frame
    

    # --------------------------------------------------
    def publish_event(self, reader_id, frame_id, zone_id, location_id, roi_polygon, obj, success, count):
        payload = {
            "readerId": reader_id,
            "type": "object_analytics",
            "attributeName": obj,
            "frameId": frame_id,
            "zoneId": zone_id,
            "locationId": location_id,
            "detectionTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "roiCoordinates": [{"x": x, "y": y} for x, y in roi_polygon] if roi_polygon else [],
            "detections": [],
            "totalCount": count,
            "success": success
        }
        publish_to_queues(payload)
        logger.info(f"[PUBLISH] {obj} | success={success} | count={count}")


