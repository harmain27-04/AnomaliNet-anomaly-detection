import sqlite3
from pathlib import Path

# ==========================================
# DATABASE PATH
# ==========================================

DATABASE_FOLDER = Path("database")
DATABASE_FOLDER.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_FOLDER / "anomalinet.db"


# ==========================================
# CONNECTION
# ==========================================

def get_connection():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


# ==========================================
# CREATE ALL TABLES
# ==========================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    # ======================================
    # USERS
    # ======================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        full_name TEXT NOT NULL,

        email TEXT UNIQUE NOT NULL,

        phone TEXT NOT NULL,

        password TEXT NOT NULL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    # ======================================
    # INCIDENTS
    # ======================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        timestamp TEXT,

        status TEXT,

        confidence REAL,

        snapshot TEXT,

        video_name TEXT

    )
    """)

    # ======================================
    # VIDEO HISTORY
    # ======================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploaded_videos(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        filename TEXT,

        uploaded_time TEXT,

        processed INTEGER DEFAULT 0

    )
    """)

    # ======================================
    # CAMERA SETTINGS
    # ======================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS camera_settings(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        camera_name TEXT,

        camera_url TEXT,

        camera_type TEXT

    )
    """)

    # ======================================
    # SYSTEM LOGS
    # ======================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_logs(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        event TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    connection.commit()

    connection.close()


# ==========================================
# LOG EVENT
# ==========================================

def log_event(event):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """
        INSERT INTO system_logs(event)

        VALUES(?)
        """,

        (event,)
    )

    connection.commit()

    connection.close()


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    initialize_database()

    print("Database Created Successfully")