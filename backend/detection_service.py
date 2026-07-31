"""
========================================================
AnomaliNet Detection Service
========================================================
"""

import os
import subprocess
import sys
from datetime import datetime
from database.db_manager import get_latest_incident
import traceback


OUTPUT_FOLDER = os.path.join(
    "frontend",
    "static",
    "processed_videos"
)

SNAPSHOT_FOLDER = os.path.join(
    "frontend",
    "static",
    "saved_incidents"
)

os.makedirs(

    OUTPUT_FOLDER,

    exist_ok=True

    )

os.makedirs(

    SNAPSHOT_FOLDER,

    exist_ok=True

    )


def run_detection(video_path):

    print("=" * 60)
    print("Starting AI Detection")
    print(video_path)
    print("=" * 60)
    

    
    
    if not os.path.exists(video_path):

        return {

            "success": False,

            "message": "Video file not found."

        }
    try:
        
        video_name = os.path.basename(video_path)

        
        output_video = os.path.join(
            OUTPUT_FOLDER,
            "output_" + video_name
        )

        print("\n" + "=" * 70)
        print("Launching AI Detection Engine...")
        print("=" * 70)

        process = subprocess.run(
            [
                sys.executable,
                "detection/real_time_detection.py",
                video_path,
                "web"
            ],
            text=True
        )

        print("=" * 70)
        print("AI Detection Engine Finished")
        print("Return Code :", process.returncode)
        print("=" * 70)

        if process.returncode != 0:

            return {

                "success": False,

                "message": process.stderr or process.stdout

                

                

                

            }
        if not os.path.exists(output_video):
            return {
                "success": False,
                "message": "Processed video not generated."
            }

        # -----------------------------

        

        

        # -----------------------------

        snapshots = []

        if os.path.exists(SNAPSHOT_FOLDER):

            snapshots = sorted(

                [

                    os.path.join(

                        SNAPSHOT_FOLDER,

                        file

                    )

                    for file in os.listdir(

                        SNAPSHOT_FOLDER

                    )

                    if file.endswith(".jpg")

                ],

                reverse=True

            )

        # -----------------------------

        

        incident = get_latest_incident()

        if incident is None:

            incident = {

                "incident_type":"Normal",

                "confidence":0,

                "alarm_status":"Not Triggered",

                "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                "snapshot_path":"",

                "video_path":"processed_videos/output_"+video_name

            }
        return {
            "success": True,
            "video_path": "processed_videos/output_" + video_name,
            "snapshots": snapshots,
            "incident": incident
        }

    except Exception as e:

        print(traceback.format_exc())

        return {

            "success": False,

            "message": str(e)

        }