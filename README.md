# Environmental Audio Recognition

Deep learning-based environmental sound classification using **Mel spectrograms and a Convolutional Neural Network (CNN)** trained on the **ESC-50 environmental sound dataset**.

The project takes an audio recording, converts it into a Mel spectrogram representation, and classifies it into one of **50 environmental sound categories**.

---

## Overview

Environmental sounds contain a wide range of acoustic patterns that can be useful for applications such as:

- Smart monitoring systems
- Urban sound analysis
- Context-aware devices
- Audio event detection
- Environmental monitoring
- Assistive technologies

This project implements an end-to-end environmental sound classification pipeline using **PyTorch**.

The system:

1. Loads environmental audio data from ESC-50.
2. Converts audio into Mel spectrograms.
3. Normalizes the spectrogram representation.
4. Trains a CNN to classify the sounds.
5. Evaluates the model using 5-fold cross-validation.
6. Performs final model training and evaluation.
7. Saves the trained model checkpoint.
8. Provides a standalone inference script for classifying external audio files.

---

## Project Objectives

The main objectives of this project are:

- Build a CNN-based environmental sound classifier.
- Use Mel spectrograms as the input representation.
- Classify sounds into the 50 ESC-50 categories.
- Evaluate the model using multiple classification metrics.
- Perform 5-fold cross-validation.
- Analyze class-level performance and confusion patterns.
- Save the final trained model for standalone inference.
- Demonstrate real-world inference using external audio files.

---

## Dataset

### ESC-50

The project uses the **ESC-50 dataset**, a collection of environmental recordings organized into 50 sound classes.

The dataset contains environmental sounds such as:

- Dog
- Cat
- Rain
- Thunderstorm
- Siren
- Airplane
- Helicopter
- Chainsaw
- Clock alarm
- Church bells
- Clapping
- Coughing
- Sneezing
- Sea waves
- Wind
- Train
- Vacuum cleaner
- Washing machine
- And many others

The dataset contains **50 classes**, with the model treating each class as a separate classification category.

The audio data is stored locally inside the project's `data/` directory.

---

## Model Architecture

The project uses a custom CNN named:

```text
ESC50CNN
```

The architecture consists of three convolutional blocks followed by global average pooling and fully connected classification layers.

```text
Input Mel Spectrogram
        │
        ▼
┌─────────────────────┐
│ Conv2D: 1 → 32      │
│ BatchNorm           │
│ ReLU                │
│ MaxPool             │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Conv2D: 32 → 64     │
│ BatchNorm           │
│ ReLU                │
│ MaxPool             │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Conv2D: 64 → 128    │
│ BatchNorm           │
│ ReLU                │
│ MaxPool             │
└─────────────────────┘
        │
        ▼
Adaptive Average Pooling
        │
        ▼
Linear: 128 → 128
        │
       ReLU
        │
     Dropout
        │
        ▼
Linear: 128 → 50
        │
        ▼
50 Sound Classes
```

### CNN Configuration

| Component | Configuration |
|---|---|
| Input channels | 1 |
| Convolution blocks | 3 |
| Conv filters | 32 → 64 → 128 |
| Kernel size | 3 × 3 |
| Padding | 1 |
| Activation | ReLU |
| Normalization | Batch Normalization |
| Pooling | Max Pooling |
| Global pooling | Adaptive Average Pooling |
| Fully connected layer | 128 units |
| Dropout | 0.3 |
| Output classes | 50 |

---

## Audio Preprocessing

The same preprocessing pipeline is used during both training and standalone inference.

### 1. Audio Loading

Audio is loaded using `librosa` with:

```text
Sample Rate: 22050 Hz
Channels: Mono
```

### 2. Fixed Duration

Each audio sample is converted to exactly **5 seconds**.

- Shorter recordings are zero-padded.
- Longer recordings are truncated.

```text
Target duration = 5 seconds
```

### 3. Mel Spectrogram

The audio is converted into a Mel spectrogram using:

| Parameter | Value |
|---|---:|
| Sample rate | 22050 Hz |
| Number of Mel bins | 128 |
| FFT size | 2048 |
| Hop length | 512 |
| Minimum frequency | 0 Hz |
| Maximum frequency | 11025 Hz |
| Power | 2.0 |

### 4. Decibel Conversion

The Mel spectrogram is converted from power values to decibels.

A maximum dynamic range of:

```text
80 dB
```

is used.

### 5. Normalization

The resulting spectrogram is normalized from:

```text
[-80, 0]
```

to:

```text
[0, 1]
```

The resulting input representation has the shape:

```text
(128, 216)
```

which becomes:

```text
(1, 128, 216)
```

after adding the CNN channel dimension.

---

## Training Methodology

The model is trained using:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 30 |
| Dropout | 0.3 |
| Loss function | Cross Entropy Loss |
| Early stopping patience | 7 |

During cross-validation, the best model state for each fold is restored based on validation loss.

---

## 5-Fold Cross-Validation

The project uses **5-fold cross-validation** to evaluate the CNN across different portions of the ESC-50 dataset.

Each fold is used as the test fold once.

### Fold Results

| Fold | Accuracy | Precision | Recall | F1 Score |
|---:|---:|---:|---:|---:|
| 1 | 37.75% | 38.44% | 37.75% | 34.55% |
| 2 | 46.50% | 47.57% | 46.50% | 44.82% |
| 3 | 43.25% | 44.82% | 43.25% | 40.63% |
| 4 | 47.25% | 47.51% | 47.25% | 44.20% |
| 5 | 44.00% | 42.43% | 44.00% | 40.32% |

### Average Cross-Validation Performance

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Test Accuracy | 43.75% | ±3.75% |
| Precision | 44.15% | ±3.84% |
| Recall | 43.75% | ±3.75% |
| F1 Score | 40.90% | ±4.09% |

The best cross-validation fold was **Fold 4**, with a test accuracy of **47.25%**.

The lowest-performing fold was **Fold 1**, with a test accuracy of **37.75%**.

---

## Final Model

After cross-validation, a final CNN was trained using the complete dataset.

The final model was saved as:

```text
models/final_environmental_sound_cnn.pth
```

The checkpoint contains:

```text
model_state_dict
class_names
class_to_idx
num_classes
```

This allows the standalone inference script to reconstruct the model and map output indices back to the corresponding ESC-50 class names.

---

## Final Model Performance

The final model achieved:

| Metric | Score |
|---|---:|
| Accuracy | **59.50%** |
| Macro Precision | **67.42%** |
| Macro Recall | **59.50%** |
| Macro F1 Score | **57.24%** |

The final evaluation was performed on **400 test samples**.

### Best Performing Classes

Several classes achieved 100% per-class accuracy in the final evaluation:

- Chainsaw
- Church bells
- Crying baby
- Clock alarm
- Thunderstorm
- Glass breaking
- Pouring water
- Rooster

### Most Difficult Classes

Some classes were significantly more difficult for the model:

- Crow — 0.00%
- Rain — 0.00%
- Can opening — 12.50%
- Fireworks — 12.50%
- Frog — 12.50%
- Wind — 12.50%
- Engine — 12.50%

This demonstrates that the model performs differently across environmental sound categories.

---

## Confusion Analysis

The confusion matrix was used to identify classes that the model frequently confused with one another.

Some of the strongest confusion patterns included:

| True Class | Predicted Class | Misclassifications |
|---|---|---:|
| Crow | Breathing | 8 |
| Frog | Pig | 4 |
| Rain | Clapping | 4 |
| Washing machine | Chainsaw | 4 |
| Wind | Airplane | 4 |
| Rain | Brushing teeth | 3 |
| Car horn | Chainsaw | 3 |
| Helicopter | Engine | 3 |
| Fireworks | Footsteps | 3 |
| Can opening | Drinking/sipping | 3 |

These errors indicate that some environmental sounds share similar acoustic characteristics, making them difficult for a relatively simple CNN to distinguish.

---

## Real-World Inference

A standalone inference script is provided:

```text
prediction.py
```

The script loads the final trained model and applies the **same preprocessing pipeline used during training**.

It accepts an external audio file and returns:

- Predicted class
- Prediction confidence
- Top-5 predictions
- Device used for inference

### Usage

Using `uv`:

```bash
uv run prediction.py path/to/audio.wav
```

For example:

```bash
uv run prediction.py data/dog.wav
```

### Example Output

```text
============================================================
        ENVIRONMENTAL SOUND CLASSIFIER
============================================================

Audio file:  data\example.wav

Prediction:  thunderstorm
Confidence:  70.75%

Top-5 Predictions:
1. thunderstorm          70.75%
2. breathing              7.64%
3. snoring                4.35%
4. airplane               3.72%
5. helicopter             3.38%

Device:      cpu
============================================================
```

The values above are an example of the inference output format. Actual predictions depend on the supplied audio file.

---

## Project Structure

```text
environmental-audio-recognition/
│
├── data/
│   └── ESC-50 audio/data files
│
├── models/
│   └── final_environmental_sound_cnn.pth
│
├── Environmental_Sound_Classification.ipynb
├── prediction.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

### Important Files

#### `Environmental_Sound_Classification.ipynb`

The main training and analysis notebook.

It contains:

- Dataset loading
- Audio preprocessing
- Mel spectrogram generation
- Dataset implementation
- CNN architecture
- Training
- Validation
- 5-fold cross-validation
- Evaluation metrics
- Confusion matrices
- Per-class analysis
- Final model training
- Final model evaluation
- Model checkpoint generation

#### `prediction.py`

Standalone inference application.

It:

- Loads the trained checkpoint
- Loads external audio
- Applies the training preprocessing pipeline
- Generates a prediction
- Calculates class probabilities
- Displays the Top-5 predictions

#### `models/final_environmental_sound_cnn.pth`

The final trained CNN checkpoint used for standalone inference.

#### `data/`

Contains the environmental sound data used by the project.

---

## Installation

### Requirements

The project uses Python and the following main libraries:

- PyTorch
- librosa
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter

The project uses **uv** for Python environment and dependency management.

### Clone the Repository

```bash
git clone <repository-url>
cd environmental-audio-recognition
```

### Create the Environment

If `uv` is installed:

```bash
uv sync
```

This installs the dependencies defined by the project configuration and lock file.

### Activate the Environment

On Windows:

```powershell
.venv\Scripts\activate
```

On Linux:

```powershell
source .venv/bin/activate
```

Alternatively, commands can be executed directly through `uv run`.

---

## Running the Inference Model

The easiest way to run the trained classifier is:

On Windows:

```powershell
uv run prediction.py "path\to\audio.wav"
```

On Linux:

```powershell
uv run prediction.py "path/to/audio.wav"
```

The model automatically uses:

```text
CUDA → if available
CPU  → otherwise
```

The current development/testing environment uses CPU inference.

---

## Training

Training and evaluation are performed through:

```text
Environmental_Sound_Classification.ipynb
```

Open the notebook using Jupyter:

```bash
uv run jupyter notebook
```

Then open:

```text
Environmental_Sound_Classification.ipynb
```

The notebook contains the complete training and evaluation workflow.

### Important

The final trained checkpoint is already included in:

```text
models/final_environmental_sound_cnn.pth
```

Therefore, **training is not required to use the standalone inference script**.

---

## Technologies Used

### Programming Language

- Python

### Deep Learning

- PyTorch

### Audio Processing

- Librosa

### Data Processing

- NumPy
- pandas

### Machine Learning Evaluation

- scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook
- uv
- Windows
- Linux
- Google Colab for model training

---

## Conclusion

This project implements an end-to-end environmental sound classification system using the ESC-50 dataset and a PyTorch CNN.

The system converts audio recordings into normalized Mel spectrograms and uses a convolutional neural network to classify them into 50 environmental sound categories.

The model achieved:

```text
Accuracy        : 59.50%
Macro Precision : 67.42%
Macro Recall    : 59.50%
Macro F1 Score  : 57.24%
```

The project also includes 5-fold cross-validation, confusion matrix analysis, per-class evaluation, final model checkpointing, and a standalone inference application.

The final `prediction.py` script makes it possible to use the trained model on external audio recordings without running the training notebook.

Overall, the project demonstrates the complete workflow of an environmental audio classification system:

```text
Environmental Audio
        ↓
Audio Preprocessing
        ↓
Mel Spectrogram
        ↓
CNN
        ↓
50-Class Classification
        ↓
Prediction + Confidence
```

---

## References

### Dataset

Piczak, K. J.  
**ESC: Dataset for Environmental Sound Classification**.  
Proceedings of the 23rd Annual ACM Conference on Multimedia, 2015.

### Libraries

- PyTorch
- Librosa
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn