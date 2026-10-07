"""
Environmental Sound Classification - Inference Script

Loads the final ESC-50 CNN checkpoint and classifies an external
audio file using the exact preprocessing pipeline used during training.

Usage:
    uv run prediction.py path/to/audio.wav

Example:
    uv run prediction.py samples/dog_bark.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "final_environmental_sound_cnn.pth"


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. AUDIO PREPROCESSING
# ============================================================
# These values match the training notebook exactly.

SAMPLE_RATE = 22050
DURATION = 5

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

FMIN = 0
FMAX = SAMPLE_RATE // 2

TOP_DB = 80


def preprocess_audio(audio_path: Path) -> torch.Tensor:
    """
    Apply the exact preprocessing pipeline used during training.

    Returns:
        Tensor with shape (1, 1, 128, 216), ready for the CNN.
    """

    # --------------------------------------------------------
    # 1. Load audio
    # --------------------------------------------------------

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # --------------------------------------------------------
    # 2. Ensure exactly 5 seconds
    # --------------------------------------------------------

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    elif len(audio) > target_length:
        audio = audio[:target_length]

    # --------------------------------------------------------
    # 3. Compute Mel Spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # --------------------------------------------------------
    # 4. Convert power to decibels
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=TOP_DB
    )

    # --------------------------------------------------------
    # 5. Normalize from [-TOP_DB, 0] to [0, 1]
    # --------------------------------------------------------

    mel_normalized = (
        mel_db + TOP_DB
    ) / TOP_DB

    # --------------------------------------------------------
    # 6. Convert to PyTorch tensor
    #
    # (128, 216)
    #      ↓
    # (1, 128, 216)
    #      ↓
    # (1, 1, 128, 216) after batch dimension
    # --------------------------------------------------------

    mel_tensor = torch.tensor(
        mel_normalized,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return mel_tensor


# ============================================================
# 4. MODEL DEFINITION
# ============================================================

class ESC50CNN(nn.Module):
    """
    CNN for ESC-50 environmental sound classification.

    This matches the architecture used in the training notebook.
    """

    def __init__(
        self,
        num_classes=50,
        dropout=0.3
    ):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x


# ============================================================
# 5. LOAD MODEL
# ============================================================

def load_model():
    """Load the final trained ESC-50 CNN checkpoint."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}\n\n"
            "Make sure models/final_environmental_sound_cnn.pth "
            "exists."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    required_keys = {
        "model_state_dict",
        "class_names",
        "class_to_idx",
        "num_classes"
    }

    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise KeyError(
            "The model checkpoint is missing required fields: "
            + ", ".join(sorted(missing_keys))
        )

    model = ESC50CNN(
        num_classes=checkpoint["num_classes"],
        dropout=0.3
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# 6. PREDICTION
# ============================================================

def predict_audio(
    audio_path: Path,
    model: nn.Module,
    class_names: list[str]
):
    """
    Predict the environmental sound in an audio file.

    Returns:
        predicted_class: Class with the highest probability.
        confidence: Probability of the predicted class.
        top_predictions: Top-5 (class, probability) pairs.
    """

    input_tensor = preprocess_audio(audio_path).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    top_k = min(5, len(class_names))

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1
    )

    top_predictions = [
        (
            class_names[index.item()],
            probability.item()
        )
        for probability, index in zip(
            top_probabilities[0],
            top_indices[0]
        )
    ]

    predicted_class = top_predictions[0][0]
    confidence = top_predictions[0][1]

    return predicted_class, confidence, top_predictions


# ============================================================
# 7. DISPLAY
# ============================================================

def print_result(
    audio_path: Path,
    predicted_class: str,
    confidence: float,
    top_predictions: list[tuple[str, float]]
):
    """Display the inference result."""

    print()
    print("=" * 60)
    print("        ENVIRONMENTAL SOUND CLASSIFIER")
    print("=" * 60)
    print()
    print(f"Audio file:  {audio_path}")
    print()
    print(f"Prediction:  {predicted_class}")
    print(f"Confidence:  {confidence * 100:.2f}%")
    print()
    print("Top-5 Predictions:")

    for rank, (class_name, probability) in enumerate(
        top_predictions,
        start=1
    ):
        print(
            f"{rank}. {class_name:<20} "
            f"{probability * 100:6.2f}%"
        )

    print()
    print(f"Device:      {DEVICE}")
    print("=" * 60)


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Classify an environmental sound using the "
            "trained ESC-50 CNN."
        )
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the audio file to classify."
    )

    return parser.parse_args()


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    args = parse_arguments()

    audio_path = args.audio.expanduser()

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    if not audio_path.is_file():
        raise ValueError(
            f"Audio path is not a file:\n{audio_path}"
        )

    print("Loading trained model...")

    model, checkpoint = load_model()

    print("Preprocessing audio...")

    (
        predicted_class,
        confidence,
        top_predictions
    ) = predict_audio(
        audio_path,
        model,
        checkpoint["class_names"]
    )

    print_result(
        audio_path,
        predicted_class,
        confidence,
        top_predictions
    )


if __name__ == "__main__":
    main()
