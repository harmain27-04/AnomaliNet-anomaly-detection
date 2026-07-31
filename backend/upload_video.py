import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {

    "mp4",

    "avi",

    "mov",

    "mkv"

}


def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(".",1)[1].lower()

        in ALLOWED_EXTENSIONS

    )


def save_uploaded_video(file):

    if file.filename == "":

        return None

    if not allowed_file(file.filename):

        return None

    os.makedirs(

        UPLOAD_FOLDER,

        exist_ok=True

    )

    filename = secure_filename(

        file.filename

    )

    filepath = os.path.join(

        UPLOAD_FOLDER,

        filename

    )

    file.save(filepath)

    return filepath