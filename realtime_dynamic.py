import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "dynamic_model.keras"
LABEL_FILE = "dynamic_labels.json"

SEQUENCE_LENGTH = 30
FEATURES = 63

CONFIDENCE_THRESHOLD = 0.70

# Number of frames with no hand required
# before another prediction is allowed.
RESET_AFTER_NO_HAND = 8

# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    print("ERROR: dynamic_model.keras not found.")
    print()
    print("Run:")
    print("python train_dynamic.py")
    exit()

if not os.path.exists(LABEL_FILE):

    print("ERROR: dynamic_labels.json not found.")
    print()
    print("Run:")
    print("python train_dynamic.py")
    exit()

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading LSTM model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABEL_FILE,
    "r",
    encoding="utf-8"
) as file:

    LABELS = json.load(file)

print("Labels:", LABELS)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)

# ============================================================
# FEATURE EXTRACTION
# Must be EXACTLY the same as collector.
# ============================================================

def extract_features(hand_landmarks):

    landmarks = []

    for landmark in hand_landmarks.landmark:

        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )

    # Wrist = origin
    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist

    # Scale normalization
    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    max_distance = np.max(distances)

    if max_distance > 0:

        landmarks = landmarks / max_distance

    return landmarks.flatten()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    hands.close()
    exit()

# ============================================================
# SEQUENCE BUFFER
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

# ============================================================
# STATE
# ============================================================

prediction = "Waiting..."
confidence = 0.0

locked = False

no_hand_count = 0

# Store recent predictions
prediction_history = deque(
    maxlen=5
)

# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read webcam.")
            break

        # Mirror
        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        # ====================================================
        # HAND DETECTED
        # ====================================================

        if results.multi_hand_landmarks:

            no_hand_count = 0

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 63 features
            features = extract_features(
                hand_landmarks
            )

            sequence.append(features)

            # =================================================
            # PREDICT
            # =================================================

            if (
                len(sequence) == SEQUENCE_LENGTH
                and not locked
            ):

                input_data = np.array(
                    sequence,
                    dtype=np.float32
                )

                input_data = np.expand_dims(
                    input_data,
                    axis=0
                )

                probabilities = model.predict(
                    input_data,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                current_confidence = float(
                    probabilities[predicted_index]
                )

                current_label = LABELS[
                    predicted_index
                ]

                # =================================================
                # CONFIDENCE CHECK
                # =================================================

                if (
                    current_confidence
                    >= CONFIDENCE_THRESHOLD
                ):

                    prediction_history.append(
                        current_label
                    )

                    # Majority vote
                    if len(prediction_history) > 0:

                        values, counts = np.unique(
                            prediction_history,
                            return_counts=True
                        )

                        majority_index = np.argmax(
                            counts
                        )

                        prediction = values[
                            majority_index
                        ]

                    confidence = (
                        current_confidence
                    )

                    # Lock after prediction
                    # until hand disappears.
                    locked = True

        # ====================================================
        # NO HAND
        # ====================================================

        else:

            no_hand_count += 1

            cv2.putText(
                frame,
                "NO HAND",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            # Clear sequence after hand disappears
            if no_hand_count >= RESET_AFTER_NO_HAND:

                sequence.clear()

                prediction_history.clear()

                locked = False

                prediction = "Waiting..."

                confidence = 0.0

        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        cv2.rectangle(
            frame,
            (15, 70),
            (500, 180),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "SIGN:",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            prediction,
            (150, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (30, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "Perform a sign - ESC to exit",
            (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "REAL-TIME DYNAMIC SIGN LANGUAGE RECOGNITION",
            frame
        )

        # ESC
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

finally:

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print()
    print("Real-time recognition stopped.")