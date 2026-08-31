import os
import threading
import webbrowser
from backend.upload_video import allowed_file
from backend.detection_service import run_detection
from backend.database import get_connection
from database.db_manager import get_all_incidents
from database.db_manager import (
    get_total_incidents,
    get_today_incidents,
    get_latest_incident,
    get_all_incidents
)
from flask import send_from_directory
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    
    flash
)

from werkzeug.utils import secure_filename

from backend.auth import register_user

from backend.login import (
    authenticate_user,
    logout_user,
    login_required,
    current_user
)

from backend.database import (
    initialize_database
)

# ==========================================
# FLASK CONFIGURATION
# ==========================================

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

app.secret_key = "AnomaliNet_Secret_Key"

# ==========================================
# UPLOAD CONFIGURATION
# ==========================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv"
}

# ==========================================
# INITIALIZE DATABASE
# ==========================================

initialize_database()

# ==========================================
# CHECK VIDEO FORMAT
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# ==========================================
# LOGIN
# ==========================================
@app.route("/processed_videos/<path:filename>")
def processed_video(filename):

    return send_from_directory(

        os.path.join(

            app.static_folder,

            "processed_videos"

        ),

        filename

    )
@app.route("/saved_incidents/<path:filename>")
def incident_image(filename):

    return send_from_directory(

        os.path.join(

            app.static_folder,

            "saved_incidents"

        ),

        filename

    )
@app.route(
    "/",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        success = authenticate_user(
            email,
            password
        )

        if success:

            flash(
                "Login Successful",
                "success"
            )

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid Email or Password",
            "danger"
        )

    return render_template(
        "login.html"
    )

# ==========================================
# REGISTER
# ==========================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        full_name = request.form["name"]

        email = request.form["email"]

        phone = request.form["phone"]

        password = request.form["password"]

        success = register_user(
            full_name,
            email,
            phone,
            password
        )

        if success:

            flash(
                "Registration Successful",
                "success"
            )

            return redirect(
                url_for("login")
            )

        flash(
            "Email already exists",
            "danger"
        )

    return render_template(
        "register.html"
    )

# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(

        "dashboard.html",

        user=current_user(),

        total_incidents=get_total_incidents(),

        today_incidents=get_today_incidents(),

        latest_incident=get_latest_incident()

    )

# ==========================================
# VIDEO UPLOAD
# ==========================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
@login_required
def upload():

    if request.method == "POST":

        if "video" not in request.files:

            flash(
                "Please choose a video.",
                "danger"
            )

            return redirect(request.url)

        video = request.files["video"]

        if video.filename == "":

            flash(
                "No file selected.",
                "danger"
            )

            return redirect(request.url)

        if allowed_file(video.filename):

            filename = secure_filename(
                video.filename
            )

            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )
            os.makedirs(
                app.config["UPLOAD_FOLDER"],
                exist_ok=True
            )
            try:

                video.save(save_path)

            except Exception as e:

                flash(
                    str(e),
                    "danger"
                )

                return redirect(
                    url_for("upload")
                )

            result = run_detection(save_path)

            

            if not result["success"]:

                flash(

                    result["message"],

                    "danger"

                )

                return redirect(url_for("upload"))

            return render_template(

                "result.html",

                result=result,

                incident=result.get("incident"),

                user=current_user()

            )
        flash(
            "Unsupported file format.",
            "danger"
        )

    return render_template(
        "upload.html"
    )

# ==========================================
# DETECTION
# ==========================================

@app.route("/detect/<filename>")
@login_required
def detect_video(filename):

    video_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )

    result = run_detection(video_path)

    if not result:

        flash(
            "Detection failed.",
            "danger"
        )

        return redirect(
            url_for("upload")
        )
    if not result.get("success"):

        flash(
            result.get("message", "Detection Failed"),
            "danger"
        )

        return redirect(url_for("upload"))

    return render_template(

        "result.html",

        result=result,

        incident=result.get("incident"),

        user=current_user()

    )

# ==========================================
# HISTORY
# ==========================================

@app.route("/history")
@login_required
def history():

    incidents = get_all_incidents()

    return render_template(

        "history.html",

        incidents=incidents

    )
# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    flash(
        "Logged Out Successfully",
        "success"
    )

    return redirect(
        url_for("login")
    )

# ==========================================
# MAIN
# ==========================================



chrome = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )