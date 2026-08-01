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
    TRACKER_CONFIG,
    TRACKER_CONFIDENCE,
    MIN_BOX_WIDTH,
    MIN_BOX_HEIGHT
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

                tracker=TRACKER_CONFIG,

                conf=TRACKER_CONFIDENCE,

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
                if person_id < 0:
                    continue

                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )
                frame_height, frame_width = frame.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(frame_width - 1, x2)
                y2 = min(frame_height - 1, y2)
                width = x2 - x1
                height = y2 - y1

                if width < MIN_BOX_WIDTH:
                    continue

                if height < MIN_BOX_HEIGHT:
                    continue
                if confidence < TRACKER_CONFIDENCE:
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
            print(f"Tracked Persons : {len(detections)}")

        return detections