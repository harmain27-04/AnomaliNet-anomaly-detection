"""
=========================================================
AnomaliNet Police Workstation
=========================================================
Live Flask dashboard for displaying anomaly incidents
received through MQTT at the Police Station.
=========================================================
"""

from flask import (
    Flask,
    render_template,
    jsonify,
    send_from_directory,
    request
)

import sqlite3
from pathlib import Path


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "police_incidents.db"

RECEIVED_INCIDENTS_FOLDER = (
    BASE_DIR / "received_incidents"
)


# ======================================================
# FLASK APPLICATION
# ======================================================

app = Flask(__name__)


# ======================================================
# DATABASE CONNECTION
# ======================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ======================================================
# PREPARE INCIDENT FOR DASHBOARD
# ======================================================

def prepare_incident(incident):

    if incident is None:

        return None

    data = dict(incident)

    snapshot = data.get("snapshot", "")

    # --------------------------------------------------
    # Convert stored filesystem path to filename
    # --------------------------------------------------

    if snapshot:

        filename = Path(snapshot).name

        data["snapshot_filename"] = filename

        data["snapshot_url"] = (
            "/snapshots/" + filename
        )

    else:

        data["snapshot_filename"] = ""

        data["snapshot_url"] = ""

    return data


# ======================================================
# GET LATEST INCIDENT
# ======================================================

def get_latest_incident():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM incidents
        ORDER BY id DESC
        LIMIT 1
        """
    )

    incident = cursor.fetchone()

    connection.close()

    return prepare_incident(incident)


# ======================================================
# GET RECENT INCIDENTS
# ======================================================

def get_recent_incidents(limit=20):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM incidents
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    incidents = cursor.fetchall()

    connection.close()

    return [
        prepare_incident(incident)
        for incident in incidents
    ]


# ======================================================
# DASHBOARD
# ======================================================

@app.route("/")
@app.route("/dashboard")
def dashboard():

    latest_incident = get_latest_incident()

    recent_incidents = get_recent_incidents()

    return render_template(

        "police_dashboard.html",

        latest_incident=latest_incident,

        recent_incidents=recent_incidents

    )


# ======================================================
# LATEST INCIDENT API
# ======================================================

@app.route("/api/latest")
def latest_api():

    incident = get_latest_incident()

    return jsonify({

        "success": True,

        "incident": incident

    })


# ======================================================
# RECENT INCIDENTS API
# ======================================================

@app.route("/api/incidents")
def incidents_api():

    incidents = get_recent_incidents()

    return jsonify({

        "success": True,

        "incidents": incidents

    })
# ======================================================
# MARK INCIDENT AS VIEWED
# ======================================================

@app.route(
    "/api/incidents/<int:incident_id>/view",
    methods=["POST"]
)
def mark_incident_viewed(incident_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE incidents
        SET status = 'VIEWED'
        WHERE id = ?
        """,
        (incident_id,)
    )

    connection.commit()

    updated = cursor.rowcount > 0

    connection.close()

    if not updated:

        return jsonify({

            "success": False,

            "message": "Incident not found."

        }), 404

    return jsonify({

        "success": True,

        "message": "Incident marked as viewed."

    })

# ======================================================
# SERVE RECEIVED SNAPSHOTS
# ======================================================

@app.route("/snapshots/<path:filename>")
def snapshot(filename):

    return send_from_directory(

        RECEIVED_INCIDENTS_FOLDER,

        filename

    )


# ======================================================
# HEALTH CHECK
# ======================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "service":
            "AnomaliNet Police Workstation",

        "database":
            str(DATABASE_PATH),

        "mqtt_snapshot_folder":
            str(RECEIVED_INCIDENTS_FOLDER)

    })


# ======================================================
# START SERVER
# ======================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "ANOMALINET POLICE WORKSTATION"
    )

    print("=" * 60)

    print(
        "Database:",
        DATABASE_PATH
    )

    print(
        "Snapshots:",
        RECEIVED_INCIDENTS_FOLDER
    )

    print("=" * 60)

    app.run(

        host="0.0.0.0",

        port=5001,

        debug=False

    )