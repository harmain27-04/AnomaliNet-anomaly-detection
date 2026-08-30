"""
========================================================
AnomaliNet Police Workstation
========================================================
Flask application for displaying incidents received
through MQTT at the Police Station.
========================================================
"""

from flask import (
    Flask,
    render_template,
    jsonify,
    send_from_directory
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
# DATABASE
# ======================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


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

    if incident:

        return dict(incident)

    return None


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
        dict(incident)
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

    if incident is None:

        return jsonify({

            "success": True,

            "incident": None

        })

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

        "service": "AnomaliNet Police Workstation",

        "database": str(DATABASE_PATH),

        "mqtt_snapshot_folder": str(
            RECEIVED_INCIDENTS_FOLDER
        )

    })


# ======================================================
# START SERVER
# ======================================================

if __name__ == "__main__":

    print("=" * 60)

    print("ANOMALINET POLICE WORKSTATION")

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

        host="127.0.0.1",

        port=5001,

        debug=False

    )