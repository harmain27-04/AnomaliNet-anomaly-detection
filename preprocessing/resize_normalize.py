import os
import cv2
from tqdm import tqdm


IMG_SIZE = (224, 224)


def resize_and_normalize(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

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

            os.makedirs(output_class_folder, exist_ok=True)

            if not os.path.exists(input_class_folder):
                print(f"Skipping missing folder: {input_class_folder}")
                continue

            videos = os.listdir(input_class_folder)

            for video in tqdm(videos, desc=f"{split}-{cls}"):

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

                frames = os.listdir(video_folder)

                for frame_name in frames:

                    frame_path = os.path.join(
                        video_folder,
                        frame_name
                    )

                    image = cv2.imread(frame_path)

                    if image is None:
                        continue

                    # Resize image
                    image = cv2.resize(
                        image,
                        IMG_SIZE
                    )

                    # Normalize image
                    image = image / 255.0

                    # Convert back for saving
                    image = (image * 255).astype("uint8")

                    save_path = os.path.join(
                        save_video_folder,
                        frame_name
                    )

                    cv2.imwrite(
                        save_path,
                        image
                    )

    print("Resize + Normalization Completed")


if __name__ == "__main__":

    input_folder = "processed_data/frames"

    output_folder = "processed_data/resized"

    resize_and_normalize(
        input_folder,
        output_folder
    )