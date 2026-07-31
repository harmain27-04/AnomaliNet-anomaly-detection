import sqlite3
from datetime import datetime

DB_PATH = "database/anomaly_logs.db"


# ===========================================
# CREATE TABLE
# ===========================================

def create_table():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        timestamp TEXT,

        incident_type TEXT,

        score REAL,

        confidence REAL,

        snapshot_path TEXT,

        video_path TEXT,

        alarm_status TEXT,

        user_email TEXT

    )
    """)

    conn.commit()

    conn.close()

    

# ===========================================
# INSERT INCIDENT
# ===========================================

def insert_incident(

    incident_type,
    score,
    confidence,
    snapshot_path,
    video_path,
    alarm_status,
    user_email

):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""

    INSERT INTO incidents(

        timestamp,
        incident_type,
        score,
        confidence,
        snapshot_path,
        video_path,
        alarm_status,
        user_email

    )

    VALUES(?,?,?,?,?,?,?,?)

    """,

    (

        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        incident_type,
        score,
        confidence,
        snapshot_path,
        video_path,
        alarm_status,
        user_email

    ))

    conn.commit()

    conn.close()


# ===========================================
# GET INCIDENTS
# ===========================================

def get_all_incidents():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *

        FROM incidents

        ORDER BY id DESC

    """)

    rows = cursor.fetchall()

    conn.close()

    return rows
def get_latest_incident():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *

        FROM incidents

        ORDER BY id DESC

        LIMIT 1

    """)

    incident = cursor.fetchone()

    conn.close()

    return incident
def get_total_incidents():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(

        "SELECT COUNT(*) FROM incidents"

    )

    total = cursor.fetchone()[0]

    conn.close()

    return total
def get_today_incidents():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""

    SELECT COUNT(*)

    FROM incidents

    WHERE date(timestamp)=date('now')

    """)

    total = cursor.fetchone()[0]

    conn.close()

    return total
