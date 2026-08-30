"""
=========================================================
AnomaliNet Configuration
=========================================================
"""

import os
import torch
from dotenv import load_dotenv
load_dotenv()
# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

# =====================================================
# MODEL PATHS
# =====================================================

YOLO_MODEL = os.path.join(
    PROJECT_ROOT,
    "yolov8n.pt"
)

LSTM_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "lstm_model",
    "lstm_model.pth"
)

AUTOENCODER_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "autoencoder",
    "autoencoder.pth"
)
# ==========================================
# TRACKER
# ==========================================

TRACKER_CONFIG = "bytetrack.yaml"

TRACKER_IOU = 0.55

TRACKER_CONFIDENCE = 0.45

MAX_PERSONS = 15

MIN_BOX_WIDTH = 40

MIN_BOX_HEIGHT = 80
# =====================================================
# VIDEO
# =====================================================

FRAME_WIDTH = 224
FRAME_HEIGHT = 224

SEQUENCE_LENGTH = 16

YOLO_CONFIDENCE = 0.50
INFERENCE_INTERVAL = 3
# =====================================================
# THRESHOLDS
# =====================================================

LSTM_THRESHOLD = 0.70

AUTOENCODER_THRESHOLD = 0.05

LSTM_WEIGHT = 0.70

AUTOENCODER_WEIGHT = 0.30

FUSION_THRESHOLD = 0.55

# =====================================================
# OUTPUT
# =====================================================

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "frontend",
    "static",
    "processed_videos"
)

SNAPSHOT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "frontend",
    "static",
    "saved_incidents"
)

LOG_FOLDER = os.path.join(
    PROJECT_ROOT,
    "logs"
)

# =====================================================
# ALERTS
# =====================================================

ENABLE_ALARM = True

ENABLE_EMAIL = False

ENABLE_TELEGRAM = False

ALARM_COOLDOWN = 5

ALARM_FILE = os.path.join(
    PROJECT_ROOT,
    "alerts",
    "alarm.wav"
)

# =====================================================
# COLORS
# =====================================================

COLOR_NORMAL = (0,255,0)

COLOR_ANOMALY = (0,0,255)

COLOR_COLLECTING = (0,255,255)


# =====================================================
# MQTT POLICE WORKSTATION
# =====================================================

# =====================================================
# MQTT POLICE WORKSTATION
# =====================================================

MQTT_ENABLED = True

# HiveMQ Cloud
MQTT_BROKER = "b5581c5800a543359895176cc38fba67.s1.eu.hivemq.cloud"

MQTT_PORT = 8883

MQTT_TOPIC = "anomalinet/incidents"

MQTT_CLIENT_ID = "anomalinet_detector"

# HiveMQ Cloud authentication
MQTT_USERNAME = os.getenv(
    "MQTT_USERNAME",
    "AnomaliNet"
)

MQTT_PASSWORD = os.getenv(
    "MQTT_PASSWORD",
    ""
)