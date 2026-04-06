import json
import pika
import pickle
import numpy as np
import cv2
import argparse
import logging
import traceback

from modules.obstacle import ObstacleDetectionSystem
from modules.seatbelt import SeatbeltSystem
from modules.phone_detection import PhoneDetectionSystem
from modules.tyre import TyreSystem
from modules.seatbelt import SeatbeltSystem
from modules.trajectory import TrajectoryDetectionSystem
from modules.oilspill import OilSpillSystem
from modules.suspended import SuspendedLoadSystem
from modules.guardrail import GuardrailSystem


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_systems(config):
    systems = []

    systems.append(ObstacleDetectionSystem(config))
    systems.append(SeatbeltSystem(config))
    systems.append(PhoneDetectionSystem(config))
    systems.append(TyreSystem(config))
    systems.append(TrajectoryDetectionSystem(config))
    systems.append(OilSpillSystem(config))
    systems.append(SuspendedLoadSystem(config))
    systems.append(GuardrailSystem(config))

    return systems


# def process_frame_all(frame, msg, systems):
#     for system in systems:
#         try:
#             frame = system.process_frame(frame, msg)
#         except Exception as e:
#             logger.error(f"{system.__class__.__name__} failed: {e}")
#     return frame
def process_frame_all(frame, msg, systems):

    camera_config = msg.get("camera_config", {})

    active_analytics = set()

    for group in camera_config.get("readerMetrics", []):
        for metric in group.get("metricJson", []):
            for st in metric.get("subType", []):
                if st.get("active"):
                    active_analytics.add(st.get("name"))

    for system in systems:
        if hasattr(system, "SUPPORTED_ANALYTICS"):
            if not active_analytics.intersection(system.SUPPORTED_ANALYTICS):
                continue  # 🔥 skip system

        try:
            frame = system.process_frame(frame, msg)
        except Exception as e:
            logger.error(f"{system.__class__.__name__} failed: {e}")

    return frame

def callback(ch, method, properties, body, systems):
    try:
        msg = pickle.loads(body)

        frame = cv2.imdecode(
            np.frombuffer(msg["frame_data"], np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            logger.error("Frame decode failed")
            return

        process_frame_all(frame, msg, systems)

    except Exception as e:
        logger.error(e)
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    systems = build_systems(config)

    connection = pika.BlockingConnection(
        pika.ConnectionParameters("localhost", heartbeat=600)
    )

    channel = connection.channel()
    queue_name = config["rabbitmq"]["read_queue"]
    channel.queue_declare(queue=queue_name, durable=True)

    logger.info("[START] Listening...")

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=lambda ch, method, props, body:
        callback(ch, method, props, body, systems),
        auto_ack=True
    )

    channel.start_consuming()


if __name__ == "__main__":
    main()