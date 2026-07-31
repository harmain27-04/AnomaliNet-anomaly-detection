import os
import cv2
import numpy as np
from tqdm import tqdm


def compute_optical_flow(
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

            if not os.path.exists(input_class_folder):
                print(
                    f"Missing folder: {input_class_folder}"
                )
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

                if len(frames) < 2:
                    continue

                prev_frame_path = os.path.join(
                    video_folder,
                    frames[0]
                )

                prev_frame = cv2.imread(
                    prev_frame_path
                )
                if prev_frame is None:
                    print(
                        f"Skipping bad video: {video_folder}"
                    )
                    continue

                prev_gray = cv2.cvtColor(
                    prev_frame,
                    cv2.COLOR_BGR2GRAY
                )

                for i in range(
                    1,
                    len(frames)
                ):

                    current_frame_path = os.path.join(
                        video_folder,
                        frames[i]
                    )

                    current_frame = cv2.imread(
                        current_frame_path
                    )
                    if current_frame is None:
                        print(
                            f"Skipping corrupted frame: {current_frame_path}"
                        )
                        continue

                    gray = cv2.cvtColor(
                        current_frame,
                        cv2.COLOR_BGR2GRAY
                    )

                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        gray,
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0
                    )

                    magnitude, angle = cv2.cartToPolar(
                        flow[..., 0],
                        flow[..., 1]
                    )

                    hsv = np.zeros_like(
                        current_frame
                    )

                    hsv[..., 1] = 255

                    hsv[..., 0] = angle * 180 / np.pi / 2

                    hsv[..., 2] = cv2.normalize(
                        magnitude,
                        None,
                        0,
                        255,
                        cv2.NORM_MINMAX
                    )

                    flow_image = cv2.cvtColor(
                        hsv,
                        cv2.COLOR_HSV2BGR
                    )

                    save_path = os.path.join(
                        save_video_folder,
                        f"flow_{i}.jpg"
                    )

                    cv2.imwrite(
                        save_path,
                        flow_image
                    )

                    prev_gray = gray

    print(
        "Optical Flow Extraction Completed"
    )


if __name__ == "__main__":

    input_folder = "processed_data/resized"

    output_folder = "processed_data/optical_flow"

    compute_optical_flow(
        input_folder,
        output_folder
    )