import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------------------------------
# Allow importing project modules
# ---------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

# ---------------------------------------
# Import Model
# ---------------------------------------

from models.lstm_model.lstm_model import LSTMModel

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SEQUENCE_LENGTH = 16

FEATURE_PATH = os.path.join(
    PROJECT_ROOT,
    "processed_data",
    "resnet_features",
    "val"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "lstm_model",
    "lstm_model.pth"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)
# ---------------------------------------
# Validation Dataset
# ---------------------------------------

class ValidationDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        classes = {
            "Fight": 1,
            "NonFight": 0
        }

        for cls, label in classes.items():

            class_path = os.path.join(
                root_dir,
                cls
            )

            if not os.path.exists(class_path):
                continue

            videos = sorted(
                os.listdir(class_path)
            )

            for video in videos:

                video_path = os.path.join(
                    class_path,
                    video
                )

                if not os.path.isdir(video_path):
                    continue

                feature_files = sorted([
                    file
                    for file in os.listdir(video_path)
                    if file.endswith(".npy")
                ])

                features = []

                for file in feature_files:

                    feature = np.load(
                        os.path.join(
                            video_path,
                            file
                        )
                    )

                    features.append(feature)

                if len(features) < SEQUENCE_LENGTH:
                    continue

                # Sliding Window
                for i in range(
                    len(features) - SEQUENCE_LENGTH + 1
                ):

                    sequence = np.array(

                        features[
                            i:i+SEQUENCE_LENGTH
                        ],

                        dtype=np.float32

                    )

                    self.samples.append(

                        (
                            sequence,
                            label,
                            video
                        )

                    )

        print("=" * 60)
        print("Validation Samples :", len(self.samples))
        print("=" * 60)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sequence, label, video = self.samples[index]

        return (

            torch.tensor(
                sequence,
                dtype=torch.float32
            ),

            torch.tensor(
                label,
                dtype=torch.long
            ),

            video

        )
    # ---------------------------------------
# Load Validation Dataset
# ---------------------------------------

validation_dataset = ValidationDataset(
    FEATURE_PATH
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# ---------------------------------------
# Load Trained LSTM Model
# ---------------------------------------

print("\nLoading LSTM Model...")

model = LSTMModel().to(DEVICE)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(

        f"\nModel not found:\n{MODEL_PATH}"

    )

checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)

model.load_state_dict(

    checkpoint

)

model.eval()

print("✓ LSTM Model Loaded Successfully")

print("=" * 60)

print("Device           :", DEVICE)

print("Validation Batch :", len(validation_loader))

print("Model Path       :", MODEL_PATH)

print("=" * 60)
# ---------------------------------------
# Evaluation
# ---------------------------------------

print("\nStarting Evaluation...\n")

all_predictions = []

all_labels = []

all_probabilities = []

all_video_names = []

correct_predictions = 0

total_predictions = 0

with torch.no_grad():

    for sequences, labels, video_names in validation_loader:

        sequences = sequences.to(DEVICE)

        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = model(sequences)

        # Softmax Probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Fight Probability
        fight_probability = probabilities[:, 1]

        # Predicted Class
        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # Store Results
        all_predictions.extend(

            predictions.cpu().numpy()

        )

        all_labels.extend(

            labels.cpu().numpy()

        )

        all_probabilities.extend(

            fight_probability.cpu().numpy()

        )

        all_video_names.extend(

            list(video_names)

        )

        correct_predictions += (

            predictions == labels

        ).sum().item()

        total_predictions += labels.size(0)

print("=" * 60)

print("Evaluation Finished")

print("Total Samples :", total_predictions)

print("Correct       :", correct_predictions)

print("Wrong         :", total_predictions - correct_predictions)

print("=" * 60)
# ---------------------------------------
# Evaluation Metrics
# ---------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

print("\n" + "=" * 60)
print("LSTM MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("=" * 60)

print("\nClassification Report\n")

print(

    classification_report(

        all_labels,
        all_predictions,

        target_names=[
            "NonFight",
            "Fight"
        ],

        digits=4

    )

)
# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(

    all_labels,

    all_predictions

)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

plt.figure(

    figsize=(7,6)

)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[

        "NonFight",

        "Fight"

    ],

    yticklabels=[

        "NonFight",

        "Fight"

    ]

)

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.title("LSTM Confusion Matrix")

confusion_path = os.path.join(

    OUTPUT_FOLDER,

    "confusion_matrix.png"

)

plt.savefig(

    confusion_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print()

print("✓ Confusion Matrix Saved")

print(confusion_path)

print("=" * 60)
# ---------------------------------------
# Save Prediction Results
# ---------------------------------------

results = []

for i in range(len(all_predictions)):

    ground_truth = (
        "Fight"
        if all_labels[i] == 1
        else "NonFight"
    )

    prediction = (
        "Fight"
        if all_predictions[i] == 1
        else "NonFight"
    )

    probability = float(
        all_probabilities[i]
    )

    correct = (
        "Correct"
        if all_predictions[i] == all_labels[i]
        else "Wrong"
    )

    results.append({

        "Video": all_video_names[i],

        "Ground Truth": ground_truth,

        "Prediction": prediction,

        "Fight Probability": round(
            probability,
            4
        ),

        "Correct": correct

    })

results_df = pd.DataFrame(results)

csv_path = os.path.join(

    OUTPUT_FOLDER,

    "prediction_results.csv"

)

results_df.to_csv(

    csv_path,

    index=False

)

print()

print("=" * 60)

print("Prediction Results Saved")

print(csv_path)

print("=" * 60)
# ---------------------------------------
# Probability Analysis
# ---------------------------------------

probabilities = np.array(all_probabilities)

print("\n" + "=" * 60)
print("FIGHT PROBABILITY ANALYSIS")
print("=" * 60)

print(f"Minimum Probability : {probabilities.min():.4f}")
print(f"Maximum Probability : {probabilities.max():.4f}")
print(f"Average Probability : {probabilities.mean():.4f}")
print(f"Median Probability  : {np.median(probabilities):.4f}")

print("=" * 60)

# ---------------------------------------
# Highest Probability Samples
# ---------------------------------------

print("\nTop 20 Highest Fight Probabilities\n")

sorted_index = np.argsort(probabilities)[::-1]

for i in sorted_index[:20]:

    print(

        f"{all_video_names[i]:35s}"

        f" GT={all_labels[i]}"

        f" Pred={all_predictions[i]}"

        f" Prob={probabilities[i]:.4f}"

    )

# ---------------------------------------
# Lowest Probability Samples
# ---------------------------------------

print("\nTop 20 Lowest Fight Probabilities\n")

sorted_index = np.argsort(probabilities)

for i in sorted_index[:20]:

    print(

        f"{all_video_names[i]:35s}"

        f" GT={all_labels[i]}"

        f" Pred={all_predictions[i]}"

        f" Prob={probabilities[i]:.4f}"

    )

# ---------------------------------------
# Final Summary
# ---------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(f"Validation Samples : {len(all_labels)}")
print(f"Accuracy           : {accuracy*100:.2f}%")
print(f"Precision          : {precision*100:.2f}%")
print(f"Recall             : {recall*100:.2f}%")
print(f"F1 Score           : {f1*100:.2f}%")

print("=" * 70)

print("\nEvaluation Completed Successfully")

print("\nGenerated Files")

print(f"1. {csv_path}")
print(f"2. {confusion_path}")

print("=" * 70)