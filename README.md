Real-Time Sign Language Recognition Using Deep Learning

A Python-based sign language recognition project that supports dynamic gesture data collection, model training, and real-time gesture prediction. The project includes saved model files and label mappings for recognizing supported gestures.

Project Overview

Sign language gestures provide an important way for people to communicate. This project explores gesture recognition using machine learning and deep learning techniques.

It includes scripts for collecting dynamic gesture data, training a model, and performing real-time recognition. The goal is to classify supported gestures from the input data processed by the application.

Note: The project recognizes only the gestures represented in its training dataset. It should not be considered a complete sign language translation system.

Features

Dynamic Gesture Data Collection: Collect gesture samples for training.
Model Training: Train a recognition model using the collected dataset.
Real-Time Recognition: Predict supported gestures using the real-time application.
Saved Models: Load trained model files for inference.
Gesture Label Mapping: Map model predictions to their corresponding gesture labels.
Reusable Workflow: Separate scripts for data collection, training, and recognition.

Technologies Used

Python
TensorFlow/Keras, if used by the training and inference scripts
NumPy
OpenCV, if used for camera input and image processing
MediaPipe, if used for hand or pose landmark extraction
Scikit-learn, if used for preprocessing or evaluation
Pickle for serialized model or data objects, if applicable
Confirm the libraries used by your source code before finalizing this list.

Project Structure

Sign-Language-Recognition/
├── collect_dynamic.py
├── train_dynamic.py
├── realtime_dynamic.py
├── data.csv
├── dynamic_data.npy
├── dynamic_labels.npy
├── dynamic_labels.json
├── dynamic_model.keras
├── gesture_model.pkl
├── requirements.txt
├── README.md
└── .gitignore

The dataset and model files shown above are optional repository assets. Include only the files required to reproduce training or run inference, and avoid committing large or sensitive data.

Installation

Prerequisites

Python installed on your system
pip package manager
A webcam, if required by the real-time recognition script
The trained model and label files required by the application

1. Create a virtual environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux or macOS

python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Use the dependency versions tested with your code. If your project uses TensorFlow, MediaPipe, or OpenCV, verify their compatibility with your Python version.

How to Run

Run the scripts from the project root directory.

Step 1: Collect gesture data

python collect_dynamic.py

This script is intended to collect dynamic gesture samples. Follow any prompts or controls implemented in the script.

Step 2: Train the model

python train_dynamic.py

This script trains the gesture recognition model using the configured dataset and saves the resulting model or supporting files, depending on its implementation.

Step 3: Start real-time recognition

python realtime_dynamic.py

If configured for webcam input, allow camera access and perform one of the gestures supported by the trained model.

Important: The exact execution sequence, input controls, output paths, and camera requirements depend on the implementation in the Python scripts. Verify these instructions against your code before publishing.

Model and Recognition Workflow

The project follows a typical gesture recognition workflow:

Collect gesture samples.

Prepare the data and associated labels.

Train a recognition model.

Save the trained model and label mappings.

Load the saved assets in the real-time recognition script.

Predict the gesture class for new input and map the prediction to a label.

The .keras extension indicates a Keras-format model file, while .pkl is commonly used for Python objects serialized with Pickle. Their exact contents and roles should be confirmed from the source code.

Dataset

The repository contains data-related files such as:

data.csv

dynamic_data.npy

dynamic_labels.npy

dynamic_labels.json

Their exact contents, sample counts, and label meanings should be documented after checking how the collection and training scripts create and load them.

If you publish sample data, ensure it contains no private information and that you have permission to share any recorded images, videos, or landmark data.

Example Input and Output

Input: A camera frame or gesture representation, depending on the configured input pipeline.

Expected output: A predicted class corresponding to one of the gestures represented in the training labels.

For example, the application might display a gesture label such as HELLO if that class exists in your dataset and the model predicts it. This is an illustrative example, not a verified prediction from the current implementation.

Add a screenshot or demonstration GIF showing an actual prediction before publishing the repository.

Model Evaluation

Evaluate the trained model using data that was not used during training. Useful metrics include:

Accuracy
Precision
Recall
F1-score
Confusion matrix

Add numerical results only after running the evaluation. Recognition performance may vary with camera position, lighting, gesture execution, background, and the range of gestures represented in the dataset.

Future Improvements

Expand the dataset with diverse gesture samples.
Evaluate performance on unseen data.
Add confusion-matrix visualization and model evaluation reports.
Improve handling of unknown or unsupported gestures.
Add a user-friendly interface and prediction confidence display, if supported by the model.
Document model architecture and inference requirements.
Explore continuous gesture recognition and sentence-level interpretation.