"""
=========================================================
AnomaliNet Person Tracker
YOLOv8 + ByteTrack
Compatible with Ultralytics 8.4.84
=========================================================
"""

from ultralytics import YOLO

from detection.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE
)


class PersonTracker:

    def __init__(self):

        print("=" * 60)
        print("Loading YOLOv8 Tracker...")
        print("=" * 60)

        self.model = YOLO(YOLO_MODEL)

        print("✓ YOLO Loaded")
        print("✓ ByteTrack Ready")

    def track(self, frame):

        detections = []

        try:

            results = self.model.track(

                source=frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=YOLO_CONFIDENCE,

                classes=[0],

                verbose=False

            )

            if not results:

                return detections

            result = results[0]

            if result.boxes is None:

                return detections

            boxes = result.boxes

            for box in boxes:

                if box.id is None:
                    continue

                person_id = int(box.id.item())

                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({

                    "id": person_id,

                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "confidence": confidence

                })

        except Exception as e:

            print()

            print("Tracker Error")

            print(e)

        return detections