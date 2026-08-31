"""
=========================================================
AnomaliNet Real-Time Anomaly Detection
Version 2.0
=========================================================
"""

import os
import sys
import cv2
import time
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

from detection.config import *

# -------------------------------------------------------
# Modules
# -------------------------------------------------------

from detection.tracker import PersonTracker
from detection.sequence_manager import SequenceManager
from detection.resnet_feature import ResNetFeatureExtractor
from detection.fusion import FeatureFusion
from detection.inference_manager import InferenceManager

# -------------------------------------------------------
# Models
# -------------------------------------------------------

from models.lstm_model.lstm_model import LSTMModel
from models.autoencoder.spatio_temporal_autoencoder import SpatioTemporalAutoencoder

# -------------------------------------------------------
# Alerts
# -------------------------------------------------------

from alerts.alert_manager import trigger_all_alerts
from alerts.mqtt_publisher import (
    publish_incident
)

# -------------------------------------------------------
# Database
# -------------------------------------------------------

from database.db_manager import insert_incident

# -------------------------------------------------------
# Device
# -------------------------------------------------------

print("=" * 70)
print("Device :", DEVICE)
print("=" * 70)

# -------------------------------------------------------
# Load LSTM
# -------------------------------------------------------

print("Loading LSTM Model...")

lstm_model = LSTMModel()

lstm_model.load_state_dict(

    torch.load(
        LSTM_MODEL,
        map_location=DEVICE
    )

)

lstm_model.to(DEVICE)

lstm_model.eval()

print("✓ LSTM Loaded")

# -------------------------------------------------------
# Load Autoencoder
# -------------------------------------------------------

print("Loading Autoencoder...")

autoencoder = SpatioTemporalAutoencoder()

autoencoder.load_state_dict(

    torch.load(
        AUTOENCODER_MODEL,
        map_location=DEVICE
    )

)

autoencoder.to(DEVICE)

autoencoder.eval()

print("✓ Autoencoder Loaded")

# -------------------------------------------------------
# Initialize Modules
# -------------------------------------------------------

tracker = PersonTracker()

extractor = ResNetFeatureExtractor()

sequence_manager = SequenceManager()

fusion = FeatureFusion()

inference_manager = InferenceManager(
    interval=INFERENCE_INTERVAL
)

print("✓ All Modules Initialized")

# -------------------------------------------------------
# Input Video
# -------------------------------------------------------

'''INPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "input_videos"
)

video_files = sorted(
    Path(INPUT_FOLDER).glob("*")
)

if len(video_files) == 0:

    raise FileNotFoundError(
        "No input video found."
    )

VIDEO_PATH = str(video_files[0])'''
# -------------------------------------------------------
# Input Video from Flask / Command Line
# -------------------------------------------------------

if len(sys.argv) < 2:

    raise ValueError(
        "Video path not provided."
    )

VIDEO_PATH = sys.argv[1]

# Optional second argument: "web" disables the OpenCV GUI when launched by Flask.
WEB_MODE = len(sys.argv) >= 3 and sys.argv[2].lower() == "web"

print("Input Video :", VIDEO_PATH)
print("Web Mode    :", WEB_MODE)

if not os.path.exists(VIDEO_PATH):

    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

print("Input Video :", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise RuntimeError(
        "Unable to open video."
    )

# -------------------------------------------------------
# Video Information
# -------------------------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

# -------------------------------------------------------
# Output Video
# -------------------------------------------------------

# -------------------------------------------------------
# Output Video
# -------------------------------------------------------

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

input_filename = os.path.basename(
    VIDEO_PATH
)

video_name_without_extension = os.path.splitext(
    input_filename
)[0]

output_filename = (
    f"output_{video_name_without_extension}.mp4"
)

output_video = os.path.join(
    OUTPUT_FOLDER,
    output_filename
)

writer = cv2.VideoWriter(

    output_video,

    cv2.VideoWriter_fourcc(*"mp4v"),

    fps,

    (width, height)

)

print("Output Video :", output_video)

# -------------------------------------------------------
# Runtime Variables
# -------------------------------------------------------

frame_number = 0

last_alert_time = 0

inactive_counter = {}

MAX_INACTIVE_FRAMES = 30

# Per-video result state. Flask reads this run-specific result instead of an older DB row.
anomaly_count = 0
max_anomaly_probability = 0.0
max_fusion_score = 0.0
best_anomaly = None
alert_count = 0

SUMMARY_FOLDER = os.path.join(PROJECT_ROOT, "frontend", "static", "detection_results")
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
summary_filename = f"summary_{video_name_without_extension}.json"
summary_path = os.path.join(SUMMARY_FOLDER, summary_filename)

def write_summary(status, message="", incident=None):

    data = {
        "success": status != "FAILED",

        "status": status,

        "message": message,

        "video_path": (
            "processed_videos/"
            + output_filename
        ),

        "video_filename": output_filename,

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "anomaly_detected": (
            status == "ANOMALY"
        ),

        "anomaly_count": int(
            anomaly_count
        ),

        "confidence": round(
            float(max_anomaly_probability),
            4
        ),

        "fusion_score": round(
            float(max_fusion_score),
            4
        ),

        "alarm_status": (
            "TRIGGERED"
            if alert_count > 0
            else
            "NOT TRIGGERED"
        ),

        "snapshot_status": (
            "CAPTURED"
            if incident
            and incident.get("snapshot_path")
            else
            "NOT REQUIRED"
        ),

        "snapshot_path": (
            incident.get("snapshot_path", "")
            if incident
            else ""
        ),

        "incident": incident
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    return data

print("=" * 70)
print("Starting Real-Time Detection...")
print("=" * 70)
# =====================================================
# Main Detection Loop
# =====================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Video Completed.")

        break

    frame_number += 1

    current_ids = set()

    try:

        # ---------------------------------------------
        # Detect & Track Persons
        # ---------------------------------------------

        detections = tracker.track(frame)

        if len(detections) == 0:

            cv2.putText(
                frame,
                "No Person Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            writer.write(frame)

            if not WEB_MODE:

                cv2.imshow(
                    "AnomaliNet",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            continue

        # ---------------------------------------------
        # Process Each Person
        # ---------------------------------------------

        for detection in detections:

            person_id = detection["id"]

            current_ids.add(person_id)

            x1, y1, x2, y2 = detection["bbox"]

            confidence = detection["confidence"]

            # -----------------------------------------
            # Keep Bounding Box Inside Frame
            # -----------------------------------------

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # -----------------------------------------
            # Extract ROI
            # -----------------------------------------

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # -----------------------------------------
            # Extract ResNet Feature
            # -----------------------------------------

            feature = extractor.extract(roi)
            if feature is None:
                continue

            if feature.shape != (2048,):
                print(f"Invalid Feature Shape : {feature.shape}")
                continue

            if feature is None:
                continue

            # -----------------------------------------
            # Store Feature
            # -----------------------------------------

            sequence_manager.add_feature(

                person_id,

                feature,
                frame_number

            )

            sequence_length = sequence_manager.length(
                person_id
            )

            # -----------------------------------------
            # Still Collecting Frames
            # -----------------------------------------

            if not sequence_manager.is_ready(person_id):

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    COLOR_COLLECTING,

                    2

                )

                cv2.putText(

                    frame,

                    f"Collecting {sequence_length}/{SEQUENCE_LENGTH}",

                    (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    COLOR_COLLECTING,

                    2

                )

                continue

            # -----------------------------------------
            # Inference Interval Check
            # -----------------------------------------

            if not inference_manager.should_run(person_id):
                continue

            # -----------------------------------------
            # Prepare Sequence
            # -----------------------------------------

            sequence = sequence_manager.get_sequence(
                person_id
            )
            if sequence is None:
                continue

            if len(sequence) != SEQUENCE_LENGTH:
                continue

            sequence_tensor = np.asarray(
                sequence,
                dtype=np.float32
            )

            if sequence_tensor.shape != (SEQUENCE_LENGTH, 2048):

                print()

                print("="*60)

                print("INVALID SEQUENCE")

                print(sequence_tensor.shape)

                print("="*60)

                continue

            sequence_tensor = torch.from_numpy(
                sequence_tensor
            )

            sequence_tensor = sequence_tensor.unsqueeze(0)

            sequence_tensor = sequence_tensor.to(DEVICE)

            # -----------------------------------------
            # LSTM Prediction
            # -----------------------------------------
            
            with torch.no_grad():
                if torch.isnan(sequence_tensor).any():

                    print("NaN detected in sequence")

                    continue

                if torch.isinf(sequence_tensor).any():

                    print("INF detected in sequence")

                    continue

                lstm_output = lstm_model(sequence_tensor)

                probabilities = torch.softmax(

                    lstm_output,

                    dim=1

                )
                if torch.isnan(probabilities).any():

                    print("NaN Probability")

                    continue

                anomaly_probability = float(
                    probabilities[0,1].item()
                )
                if anomaly_probability < 0:

                    anomaly_probability = 0

                if anomaly_probability > 1:

                    anomaly_probability = 1
                print()

                print("="*60)

                print("LSTM INPUT")

                print("="*60)

                print("Shape :", sequence_tensor.shape)

                print("Min   :", float(sequence_tensor.min()))

                print("Max   :", float(sequence_tensor.max()))

                print("Mean  :", float(sequence_tensor.mean()))

                print("="*60)

            # -----------------------------------------
            # Autoencoder Prediction
            # -----------------------------------------

            with torch.no_grad():

                # -----------------------------------------
# Autoencoder Input
# -----------------------------------------

                ae_input = sequence_tensor.mean(dim=1)

                if ae_input.shape != (1, 2048):

                    print()

                    print("=" * 60)

                    print("INVALID AE INPUT")

                    print(ae_input.shape)

                    print("=" * 60)

                    continue

                reconstructed = autoencoder(ae_input)

                reconstruction_error = torch.nn.functional.mse_loss(

                    reconstructed,

                    ae_input,

                    reduction="mean"

                ).item()

                if np.isnan(reconstruction_error):

                    print("Invalid Reconstruction Error")

                    continue

            # -----------------------------------------
            # Fusion
            # -----------------------------------------
            if not (0.0 <= anomaly_probability <= 1.0):

                print("Invalid LSTM Probability")

                continue

            if reconstruction_error < 0:

                print("Invalid Reconstruction Error")

                continue
            fusion_result = fusion.classify(

                anomaly_probability,

                reconstruction_error

            )

            fusion_score = fusion_result["fusion_score"]

            is_anomaly = fusion_result["is_anomaly"]
            print("\n" + "=" * 60)
            print(f"Frame              : {frame_number}")
            print(f"Person ID          : {person_id}")
            print(f"LSTM Probability   : {anomaly_probability:.4f}")
            print(f"AE Reconstruction  : {reconstruction_error:.6f}")
            print(f"LSTM Weight        : {LSTM_WEIGHT}")
            print(f"AE Weight          : {AUTOENCODER_WEIGHT}")
            print(f"Fusion Threshold   : {FUSION_THRESHOLD}")
            print(f"Fusion Score       : {fusion_score:.4f}")
            print(f"Is Anomaly         : {is_anomaly}")
            print("=" * 60)

            # -----------------------------------------
            # Label
            # -----------------------------------------

            if is_anomaly:

                label = "ANOMALY"

                color = COLOR_ANOMALY

            else:

                label = "NORMAL"

                color = COLOR_NORMAL

            # -----------------------------------------
            # Draw Bounding Box
            # -----------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                color,

                2

            )

            cv2.putText(

                frame,

                f"ID:{person_id}",

                (x1, y1 - 65),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                color,

                2

            )

            cv2.putText(

                frame,

                f"{label}",

                (x1, y1 - 45),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                color,

                2

            )

            cv2.putText(

                frame,

                f"LSTM : {anomaly_probability:.2f}",

                (x1, y1 - 25),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                color,

                2

            )

            cv2.putText(

                frame,

                f"Fusion : {fusion_score:.2f}",

                (x1, y1 - 5),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                color,

                2

            )

            # -------- Part 3 starts here --------
                        # -----------------------------------------
            # Alert Cooldown
            # -----------------------------------------

            if is_anomaly:
                anomaly_count += 1
                max_anomaly_probability = max(max_anomaly_probability, anomaly_probability)
                max_fusion_score = max(max_fusion_score, fusion_score)

            current_time = time.time()

            if (

                is_anomaly

                and

                (current_time - last_alert_time >= ALARM_COOLDOWN)

            ):

                print("=" * 60)
                print("ANOMALY DETECTED")
                print("=" * 60)

                snapshot = trigger_all_alerts(frame)
                # -----------------------------------------
                # Publish Incident to Police Workstation
                # -----------------------------------------

                mqtt_success = publish_incident(

                    person_id=person_id,

                    confidence=anomaly_probability,

                    fusion_score=fusion_score,

                    snapshot_path=snapshot,

                    camera_id="CAMERA_01"

                )

                if mqtt_success:

                    print(
                        "Incident sent to Police Workstation"
                    )

                else:

                    print(
                        "Failed to send incident through MQTT"
                    )

                try:

                    insert_incident(

                        incident_type="Anomaly",
                        score=float(fusion_score),
                        confidence=float(anomaly_probability),
                        snapshot_path=snapshot if snapshot else "",
                        video_path=output_video,
                        alarm_status="Triggered",
                        user_email=""
                        

                    )
                    print()

                    print("="*60)

                    print("INCIDENT SAVED")

                    print("Score :", fusion_score)

                    print("Confidence :", anomaly_probability)

                    print("="*60)

                except Exception as e:

                    print()

                    print("="*60)

                    print("DATABASE ERROR")

                    print(type(e).__name__)

                    print(e)

                    print("="*60)

                alert_count += 1
                best_anomaly = {
                    "person_id": int(person_id),
                    "confidence": round(float(anomaly_probability), 4),
                    "fusion_score": round(float(fusion_score), 4),
                    "snapshot_path": snapshot if snapshot else "",
                    "alarm_status": "Triggered",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mqtt_sent": bool(mqtt_success)
                }
                last_alert_time = current_time

        # ---------------------------------------------
        # Remove Missing Persons
        # ---------------------------------------------

        missing_ids = set(

            inactive_counter.keys()

        ) | set(

            sequence_manager.buffers.keys()

        )

        for pid in missing_ids:

            if pid in current_ids:

                inactive_counter[pid] = 0

                continue

            inactive_counter[pid] = (

                inactive_counter.get(pid, 0)

                + 1

            )

            if (

                inactive_counter[pid]

                >=

                MAX_INACTIVE_FRAMES

            ):

                print(

                    f"Removing Person {pid}"

                )

                sequence_manager.remove(pid)

                inference_manager.remove(pid)

                inactive_counter.pop(

                    pid,

                    None

                )

    except Exception as e:

        print()

        print("=" * 60)

        print("Frame Processing Error")

        print(e)

        print("=" * 60)

    # -------------------------------------------------
    # Display Information
    # -------------------------------------------------

    cv2.putText(

        frame,

        f"Frame : {frame_number}",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255,255,255),

        2

    )

    # -------------------------------------------------
    # Save Output
    # -------------------------------------------------

    writer.write(frame)

    # -------------------------------------------------
    # Display Window
    # -------------------------------------------------

    if not WEB_MODE:
        cv2.imshow(
            "AnomaliNet",
            frame
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break
    sequence_manager.cleanup(
        frame_number
    )

# =====================================================
# Cleanup
# =====================================================

print()

print("=" * 70)

print("Cleaning Resources...")

print("=" * 70)

cap.release()

writer.release()

if not WEB_MODE:
    cv2.destroyAllWindows()


# =====================================================
# Final Detection Result
# =====================================================

if anomaly_count > 0:

    final_status = "ANOMALY"

    final_message = (
        "Abnormal activity detected."
    )

    final_incident = best_anomaly

else:

    final_status = "NORMAL"

    final_message = (
        "No abnormal activity detected."
    )

    final_incident = None


final_result = write_summary(

    status=final_status,

    message=final_message,

    incident=final_incident

)

print()
print("=" * 70)
print("FINAL DETECTION RESULT")
print("=" * 70)

print(
    "Status          :",
    final_result["status"]
)

print(
    "Confidence      :",
    final_result["confidence"]
)

print(
    "Fusion Score    :",
    final_result["fusion_score"]
)

print(
    "Alarm           :",
    final_result["alarm_status"]
)

print(
    "Snapshot        :",
    final_result["snapshot_status"]
)

print(
    "Summary File    :",
    summary_path
)

print(
    "Processed Video :",
    output_video
)

print("=" * 70)
sequence_manager.clear()

inference_manager.clear()

print("Detection Finished.")

print("Processed Video Saved At:")

print(output_video)