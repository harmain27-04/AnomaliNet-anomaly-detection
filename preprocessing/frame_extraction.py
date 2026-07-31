import cv2
import os
from tqdm import tqdm


def extract_frames(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_name = os.path.join(
            output_folder,
            f"frame_{frame_count:04d}.jpg"
        )

        cv2.imwrite(frame_name, frame)

        frame_count += 1

    cap.release()

    print(f"Extracted {frame_count} frames")


def process_dataset(dataset_path, save_path):

    classes = ["Fight", "NonFight"]

    for split in ["train", "val"]:

        for cls in classes:

            video_folder = os.path.join(
                dataset_path,
                split,
                cls
            )

            save_folder = os.path.join(
                save_path,
                split,
                cls
            )

            os.makedirs(save_folder, exist_ok=True)

            videos = os.listdir(video_folder)

            for video in tqdm(videos):

                video_path = os.path.join(
                    video_folder,
                    video
                )

                video_name = f"vid_{videos.index(video)}"

                output_folder = os.path.join(
                    save_folder,
                    video_name
                )

# Create short path to avoid Windows path-too-long error
                output_folder = output_folder[:150]

                extract_frames(
                    video_path,
                    output_folder
                )


if __name__ == "__main__":

    dataset_path = "dataset/rwf2000"

    save_path = "processed_data/frames"

    process_dataset(
        dataset_path,
        save_path
    )