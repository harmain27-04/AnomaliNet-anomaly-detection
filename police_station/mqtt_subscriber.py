"""
=========================================================
AnomaliNet Police Station MQTT Subscriber
=========================================================
Receives anomaly incidents from the MQTT broker and
stores them in the Police Station database.
=========================================================
"""

import json
import base64
import os
import ssl
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

from police_station.police_database import (
    initialize_database,
    insert_incident
)
from police_station.police_alarm import trigger_police_alarm

# =====================================================
# MQTT CONFIGURATION
# =====================================================

# =====================================================
# MQTT CONFIGURATION
# =====================================================

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")


# =====================================================
# SNAPSHOT FOLDER
# =====================================================

SNAPSHOT_FOLDER = os.path.join(
    "police_station",
    "received_incidents"
)

os.makedirs(
    SNAPSHOT_FOLDER,
    exist_ok=True
)


# =====================================================
# CONNECT CALLBACK
# =====================================================

def on_connect(
    client,
    userdata,
    flags,
    rc
):

    print("=" * 60)

    print(
        "POLICE STATION MQTT SUBSCRIBER"
    )

    print("=" * 60)

    if rc == 0:

        print(
            "Connected to MQTT Broker"
        )

        print(
            "Broker:",
            MQTT_BROKER
        )

        print(
            "Port:",
            MQTT_PORT
        )

        print(
            "Topic:",
            MQTT_TOPIC
        )

        print("=" * 60)

        client.subscribe(
            MQTT_TOPIC,
            qos=1
        )

        print(
            "Waiting for incidents..."
        )

        print("=" * 60)

    else:

        print(
            "MQTT Connection Failed"
        )

        print(
            "Return Code:",
            rc
        )


# =====================================================
# MESSAGE CALLBACK
# =====================================================

def on_message(
    client,
    userdata,
    message
):

    print()
    print("=" * 60)

    print(
        "NEW INCIDENT RECEIVED"
    )

    print("=" * 60)

    try:

        # -------------------------------------------------
        # Decode MQTT payload
        # -------------------------------------------------

        payload = message.payload.decode(
            "utf-8"
        )

        incident = json.loads(
            payload
        )

        # -------------------------------------------------
        # Extract incident information
        # -------------------------------------------------

        incident_type = incident.get(
            "incident_type",
            "UNKNOWN"
        )

        person_id = incident.get(
            "person_id"
        )
        # -------------------------------------------------
        # Police Alarm
        # -------------------------------------------------
        
        camera_id = incident.get(
            "camera_id",
            "UNKNOWN"
        )

        confidence = incident.get(
            "confidence",
            0
        )

        fusion_score = incident.get(
            "fusion_score",
            0
        )

        timestamp = incident.get(
            "timestamp",
            ""
        )

        snapshot_data = incident.get(
            "snapshot"
        )

        # -------------------------------------------------
        # Print incident information
        # -------------------------------------------------

        print(
            "Incident Type :",
            incident_type
        )

        print(
            "Person ID     :",
            person_id
        )

        print(
            "Camera ID     :",
            camera_id
        )

        print(
            "Confidence    :",
            confidence
        )

        print(
            "Fusion Score  :",
            fusion_score
        )

        print(
            "Timestamp     :",
            timestamp
        )
        # -------------------------------------------------
        # Trigger Police Station Alarm
        # -------------------------------------------------

        print()
        print("=" * 60)
        print("🚨 ANOMALY INCIDENT RECEIVED")
        print("=" * 60)

        try:

            trigger_police_alarm()

            print(
                "🚨 POLICE ALARM TRIGGERED"
            )

        except Exception as e:

            print(
                "POLICE ALARM ERROR:"
            )

            print(
                type(e).__name__
            )

            print(
                e
            )

        print("=" * 60)
        # -------------------------------------------------
        # Save snapshot
        # -------------------------------------------------

        snapshot_path = ""

        if snapshot_data:

            try:

                image_bytes = base64.b64decode(
                    snapshot_data
                )

                safe_timestamp = (
                    timestamp
                    .replace(
                        ":",
                        "-"
                    )
                    .replace(
                        " ",
                        "_"
                    )
                )

                snapshot_filename = (
                    f"incident_"
                    f"{person_id}_"
                    f"{safe_timestamp}.jpg"
                )

                snapshot_path = os.path.join(
                    SNAPSHOT_FOLDER,
                    snapshot_filename
                )

                with open(
                    snapshot_path,
                    "wb"
                ) as image_file:

                    image_file.write(
                        image_bytes
                    )

                print(
                    "Snapshot Saved:"
                )

                print(
                    snapshot_path
                )

            except Exception as e:

                print(
                    "Snapshot Decode Error:",
                    e
                )

        else:

            print(
                "No Snapshot Received"
            )

        # -------------------------------------------------
        # Save incident to Police Database
        # -------------------------------------------------

        insert_incident(

            incident_type=incident_type,

            person_id=person_id,

            camera_id=camera_id,

            confidence=float(
                confidence
            ),

            fusion_score=float(
                fusion_score
            ),

            timestamp=timestamp,

            snapshot=snapshot_path

        )

        print()
        print(
            "INCIDENT SAVED TO POLICE DATABASE"
        )

        print("=" * 60)

    except json.JSONDecodeError as e:

        print(
            "Invalid MQTT JSON:"
        )

        print(e)

    except Exception as e:

        print(
            "MQTT Message Processing Error:"
        )

        print(
            type(e).__name__
        )

        print(
            e
        )

        print("=" * 60)


# =====================================================
# MAIN
# =====================================================

def start_subscriber():

    # -------------------------------------------------
    # Make sure database exists
    # -------------------------------------------------

    initialize_database()

    # -------------------------------------------------
    # Create MQTT client
    # -------------------------------------------------

    client = mqtt.Client(
        client_id="anomalinet_police_station"
    )
    # -------------------------------------------------
    # HiveMQ Cloud authentication
    # -------------------------------------------------

    client.username_pw_set(
        MQTT_USERNAME,
        MQTT_PASSWORD
    )

    # -------------------------------------------------
    # TLS encryption
    # -------------------------------------------------

    client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED
    )

    # -------------------------------------------------
    # Register callbacks
    # -------------------------------------------------

    client.on_connect = on_connect

    client.on_message = on_message

    # -------------------------------------------------
    # Connect to broker
    # -------------------------------------------------

    print()
    print(
        "Connecting to MQTT Broker..."
    )

    client.connect(

        MQTT_BROKER,

        MQTT_PORT,

        10

    )

    # -------------------------------------------------
    # Keep listening
    # -------------------------------------------------

    client.loop_forever()


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    start_subscriber()