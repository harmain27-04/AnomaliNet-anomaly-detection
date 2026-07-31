import os
import numpy as np
from tqdm import tqdm

SEQUENCE_LENGTH = 16


def create_sequences(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    splits = ["train", "val"]
    classes = ["Fight", "NonFight"]

    for split in splits:
        for cls in classes:

            source_path = os.path.join(
                input_folder,
                split,
                cls
            )

            save_path = os.path.join(
                output_folder,
                split,
                cls
            )

            os.makedirs(save_path, exist_ok=True)

            if not os.path.exists(source_path):
                continue

            videos = os.listdir(source_path)

            for video in tqdm(videos,
                              desc=f"{split}-{cls}"):

                video_folder = os.path.join(
                    source_path,
                    video
                )

                frames = sorted(
                    os.listdir(video_folder)
                )

                sequence_count = 0

                for i in range(
                    0,
                    len(frames)-SEQUENCE_LENGTH
                ):

                    sequence = []

                    for j in range(SEQUENCE_LENGTH):

                        frame_path = os.path.join(
                            video_folder,
                            frames[i+j]
                        )

                        sequence.append(frame_path)

                    save_file = os.path.join(
                        save_path,
                        f"{video}_{sequence_count}.npy"
                    )

                    np.save(
                        save_file,
                        np.array(sequence)
                    )

                    sequence_count += 1

    print("Sequence Generation Completed")


if __name__ == "__main__":

    input_folder = "processed_data/roi_frames"

    output_folder = "processed_data/sequences"

    create_sequences(
        input_folder,
        output_folder
    )