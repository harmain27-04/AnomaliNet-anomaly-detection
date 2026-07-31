import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm


# Load YOLOv8 model
model = YOLO("yolov8n.pt")


def extract_roi(
    input_folder,
    output_folder
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    splits = ["train", "val"]
    classes = ["Fight", "NonFight"]

    for split in splits:
        for cls in classes:

            input_class_folder = os.path.join(
                input_folder,
                split,
                cls
            )

            output_class_folder = os.path.join(
                output_folder,
                split,
                cls
            )

            os.makedirs(
                output_class_folder,
                exist_ok=True
            )

            if not os.path.exists(
                input_class_folder
            ):
                continue

            videos = os.listdir(
                input_class_folder
            )

            for video in tqdm(
                videos,
                desc=f"{split}-{cls}"
            ):

                video_folder = os.path.join(
                    input_class_folder,
                    video
                )

                save_video_folder = os.path.join(
                    output_class_folder,
                    video
                )

                os.makedirs(
                    save_video_folder,
                    exist_ok=True
                )

                frames = sorted(
                    os.listdir(video_folder)
                )

                for frame in frames:

                    frame_path = os.path.join(
                        video_folder,
                        frame
                    )

                    image = cv2.imread(
                        frame_path
                    )

                    if image is None:
                        continue

                    results = model(
                        image,
                        verbose=False
                    )

                    detected = False

                    for result in results:

                        for box in result.boxes:

                            cls_id = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            # Person class in COCO
                            if (
                                cls_id == 0
                                and confidence > 0.5
                            ):

                                x1, y1, x2, y2 = map(
                                    int,
                                    box.xyxy[0]
                                )

                                roi = image[
                                    y1:y2,
                                    x1:x2
                                ]

                                if roi.size == 0:
                                    continue

                                roi = cv2.resize(
                                    roi,
                                    (224, 224)
                                )

                                save_path = os.path.join(
                                    save_video_folder,
                                    frame
                                )

                                cv2.imwrite(
                                    save_path,
                                    roi
                                )

                                detected = True
                                break

                        if detected:
                            break

                    # If no human detected
                    if not detected:

                        image = cv2.resize(
                            image,
                            (224, 224)
                        )

                        save_path = os.path.join(
                            save_video_folder,
                            frame
                        )

                        cv2.imwrite(
                            save_path,
                            image
                        )

    print(
        "ROI Extraction Completed"
    )


if __name__ == "__main__":

    input_folder = "processed_data/resized"

    output_folder = "processed_data/roi_frames"

    extract_roi(
        input_folder,
        output_folder
    )