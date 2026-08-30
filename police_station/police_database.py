"""
=========================================================
AnomaliNet Police Station Database
=========================================================
Stores incidents received from the AnomaliNet MQTT broker.
=========================================================
"""

import sqlite3
from pathlib import Path


# =====================================================
# DATABASE PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "police_incidents.db"


# =====================================================
# DATABASE CONNECTION
# =====================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# =====================================================
# CREATE TABLE
# =====================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            incident_type TEXT,

            person_id INTEGER,

            camera_id TEXT,

            confidence REAL,

            fusion_score REAL,

            timestamp TEXT,

            snapshot TEXT,

            status TEXT DEFAULT 'NEW'

        )
    """)

    connection.commit()

    connection.close()

    print(
        "Police Station Database Ready"
    )


# =====================================================
# INSERT INCIDENT
# =====================================================

def insert_incident(

    incident_type,

    person_id,

    camera_id,

    confidence,

    fusion_score,

    timestamp,

    snapshot

):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO incidents(

            incident_type,
            person_id,
            camera_id,
            confidence,
            fusion_score,
            timestamp,
            snapshot,
            status

        )

        VALUES(?,?,?,?,?,?,?,?)
    """,

    (

        incident_type,

        person_id,

        camera_id,

        confidence,

        fusion_score,

        timestamp,

        snapshot,

        "NEW"

    ))

    connection.commit()

    connection.close()


# =====================================================
# GET ALL INCIDENTS
# =====================================================

def get_all_incidents():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM incidents
        ORDER BY id DESC
    """)

    incidents = cursor.fetchall()

    connection.close()

    return incidents


# =====================================================
# GET LATEST INCIDENT
# =====================================================

def get_latest_incident():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM incidents
        ORDER BY id DESC
        LIMIT 1
    """)

    incident = cursor.fetchone()

    connection.close()

    return incident


# =====================================================
# TEST DATABASE
# =====================================================

if __name__ == "__main__":

    initialize_database()

    print(
        "Database Path:"
    )

    print(
        DATABASE_PATH
    )