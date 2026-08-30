"""
=========================================================
AnomaliNet MQTT Incident Publisher
=========================================================
Publishes confirmed anomaly incidents to the
Police Workstation through an MQTT broker.
=========================================================
"""

import json
import base64
import ssl
from datetime import datetime

import paho.mqtt.client as mqtt

from detection.config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    MQTT_CLIENT_ID,
    MQTT_ENABLED,
    MQTT_USERNAME,
    MQTT_PASSWORD
)


def encode_snapshot(snapshot_path):

    if not snapshot_path:
        return None

    try:

        with open(
            snapshot_path,
            "rb"
        ) as image_file:

            encoded_image = base64.b64encode(
                image_file.read()
            ).decode(
                "utf-8"
            )

        return encoded_image

    except Exception as e:

        print(
            "Snapshot Encoding Error:",
            e
        )

        return None


def publish_incident(

    person_id,

    confidence,

    fusion_score,

    snapshot_path,

    camera_id="CAMERA_01"

):

    if not MQTT_ENABLED:

        print(
            "MQTT is disabled."
        )

        return False

    try:

        snapshot_data = encode_snapshot(
            snapshot_path
        )

        incident = {

            "incident_type":
                "ANOMALY",

            "person_id":
                int(person_id),

            "camera_id":
                camera_id,

            "confidence":
                round(
                    float(confidence),
                    4
                ),

            "fusion_score":
                round(
                    float(fusion_score),
                    4
                ),

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "snapshot":
                snapshot_data

        }

        payload = json.dumps(
            incident
        )

        client = mqtt.Client(

            client_id=(
                f"{MQTT_CLIENT_ID}_"
                f"{int(datetime.now().timestamp())}"
            )

        )

        # =============================================
        # HiveMQ Cloud Authentication
        # =============================================

        client.username_pw_set(
            MQTT_USERNAME,
            MQTT_PASSWORD
        )

        # =============================================
        # TLS Encryption
        # =============================================

        client.tls_set(
            cert_reqs=ssl.CERT_REQUIRED
        )

        # =============================================
        # Connect to HiveMQ Cloud
        # =============================================

        client.connect(

            MQTT_BROKER,

            MQTT_PORT,

            10

        )

        client.loop_start()

        # =============================================
        # Publish Incident
        # =============================================

        result = client.publish(

            MQTT_TOPIC,

            payload,

            qos=1

        )

        result.wait_for_publish()

        client.loop_stop()

        client.disconnect()

        print("=" * 60)

        print(
            "MQTT INCIDENT PUBLISHED"
        )

        print(
            "Broker:",
            MQTT_BROKER
        )

        print(
            "Topic:",
            MQTT_TOPIC
        )

        print(
            "Person ID:",
            person_id
        )

        print(
            "Confidence:",
            confidence
        )

        print(
            "Fusion Score:",
            fusion_score
        )

        print("=" * 60)

        return True

    except Exception as e:

        print("=" * 60)

        print(
            "MQTT PUBLISH ERROR"
        )

        print(
            type(e).__name__
        )

        print(
            e
        )

        print("=" * 60)

        return False