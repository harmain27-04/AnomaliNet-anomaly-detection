import cv2
import os
from datetime import datetime


SAVE_FOLDER = os.path.join(

    "frontend",

    "static",

    "saved_incidents"

)


os.makedirs(

    SAVE_FOLDER,

    exist_ok=True

)


def save_snapshot(frame):

    filename = datetime.now().strftime(

        "%Y%m%d_%H%M%S_%f.jpg"

    )

    path = os.path.join(

        SAVE_FOLDER,

        filename

    )

    success = cv2.imwrite(

        path,

        frame

    )

    if success:

        print("Snapshot Saved :", path)

        return path

    print("Snapshot Save Failed")

    return None