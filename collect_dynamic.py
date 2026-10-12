import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30
DATA_FILE = "dynamic_data.npy"
LABEL_FILE = "dynamic_labels.npy"

LABELS = [
    "HELLO",
    "YES",
    "NO",
    "HELP",
    "THANKYOU",
    "PLEASE",
    "SORRY",
    "GOOD",
    "BAD"
]

# Number of sequences to collect for EACH sign
TARGET_PER_SIGN = 30

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
# 21 landmarks x 3 = 63 features
# Wrist becomes origin.
# Scale normalization makes it less sensitive to hand distance.
# ============================================================

def extract_features(hand_landmarks):

    landmarks = []

    for landmark in hand_landmarks.landmark:
        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    landmarks = np.array(landmarks, dtype=np.float32)

    # Wrist is landmark 0
    wrist = landmarks[0].copy()

    # Move wrist to origin
    landmarks = landmarks - wrist

    # Scale normalization
    distances = np.linalg.norm(landmarks, axis=1)

    max_distance = np.max(distances)

    if max_distance > 0:
        landmarks = landmarks / max_distance

    return landmarks.flatten()


# ============================================================
# LOAD OLD DATA IF IT EXISTS
# ============================================================

if os.path.exists(DATA_FILE) and os.path.exists(LABEL_FILE):

    try:
        data = np.load(DATA_FILE)
        labels = np.load(LABEL_FILE)

        # Check old dataset shape
        if (
            len(data.shape) == 3
            and data.shape[1] == SEQUENCE_LENGTH
            and data.shape[2] == 63
            and len(data) == len(labels)
        ):
            sequences = data.tolist()
            sequence_labels = labels.tolist()

            print("Existing compatible dataset loaded.")
            print("Sequences:", len(sequences))

        else:
            print("Old dataset format is incompatible.")
            print("Starting a NEW dataset.")

            sequences = []
            sequence_labels = []

    except Exception:
        print("Could not load old dataset.")
        print("Starting a NEW dataset.")

        sequences = []
        sequence_labels = []

else:

    print("Starting a NEW dataset.")

    sequences = []
    sequence_labels = []


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    hands.close()
    exit()


# ============================================================
# COUNTS
# ============================================================

def get_counts():

    counts = {}

    for label in LABELS:
        counts[label] = 0

    for value in sequence_labels:

        if 0 <= int(value) < len(LABELS):
            counts[LABELS[int(value)]] += 1

    return counts


# ============================================================
# PRINT MENU
# ============================================================

print()
print("=" * 55)
print("       9-SIGN DYNAMIC DATA COLLECTION")
print("=" * 55)

for i, label in enumerate(LABELS, start=1):
    print(f"{i} = {label}")

print()
print(f"Target: {TARGET_PER_SIGN} sequences per sign")
print(f"Total target: {TARGET_PER_SIGN * len(LABELS)} sequences")
print()
print("Press 1-9 to select and START recording.")
print("ESC = Quit")
print("=" * 55)
print()

# ============================================================
# MAIN LOOP
# ============================================================

selected_label = None
recording = False
sequence = []

countdown_start = None
countdown_seconds = 2

# Current frame counter
frame_count = 0

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        # Draw hand
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # ====================================================
        # KEY INPUT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:
            break

        # ====================================================
        # SELECT SIGN
        # ====================================================

        if not recording and 49 <= key <= 57:

            index = key - 49

            if index < len(LABELS):

                selected_label = index

                print()
                print("=" * 40)
                print("Selected:", LABELS[selected_label])
                print("Get ready...")
                print("=" * 40)

                countdown_start = time.time()
                recording = "countdown"

                sequence = []

        # ====================================================
        # COUNTDOWN
        # ====================================================

        if recording == "countdown":

            elapsed = time.time() - countdown_start

            remaining = countdown_seconds - elapsed

            if remaining > 0:

                countdown_text = f"GET READY: {int(remaining) + 1}"

                cv2.putText(
                    frame,
                    countdown_text,
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 255),
                    3
                )

            else:

                recording = True
                sequence = []
                frame_count = 0

                print("Recording", LABELS[selected_label], "...")

        # ====================================================
        # RECORD 30 FRAMES
        # ====================================================

        if recording is True:

            # Extract hand features
            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                features = extract_features(hand_landmarks)

                sequence.append(features)

            else:

                # If hand temporarily disappears,
                # use zero vector for this frame.
                sequence.append(
                    np.zeros(63, dtype=np.float32)
                )

            frame_count = len(sequence)

            cv2.putText(
                frame,
                f"RECORDING: {LABELS[selected_label]}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {frame_count}/{SEQUENCE_LENGTH}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Finished
            if len(sequence) >= SEQUENCE_LENGTH:

                sequence_array = np.array(
                    sequence,
                    dtype=np.float32
                )

                # Safety check
                if sequence_array.shape == (30, 63):

                    sequences.append(sequence_array)
                    sequence_labels.append(selected_label)

                    print(
                        "SAVED:",
                        LABELS[selected_label]
                    )

                    counts = get_counts()

                    print(
                        f"{LABELS[selected_label]}: "
                        f"{counts[LABELS[selected_label]]}/"
                        f"{TARGET_PER_SIGN}"
                    )

                    print(
                        "Total sequences:",
                        len(sequences)
                    )

                else:

                    print(
                        "ERROR: Invalid sequence shape:",
                        sequence_array.shape
                    )

                # Reset
                recording = False
                selected_label = None
                sequence = []

        # ====================================================
        # DISPLAY COUNTS
        # ====================================================

        counts = get_counts()

        y = 140

        for label in LABELS:

            text = f"{label}: {counts[label]}/{TARGET_PER_SIGN}"

            cv2.putText(
                frame,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            y += 25

        cv2.imshow(
            "Dynamic Sign Data Collection",
            frame
        )

finally:

    # ========================================================
    # SAVE DATA
    # ========================================================

    if len(sequences) > 0:

        data_array = np.array(
            sequences,
            dtype=np.float32
        )

        labels_array = np.array(
            sequence_labels,
            dtype=np.int32
        )

        np.save(DATA_FILE, data_array)
        np.save(LABEL_FILE, labels_array)

        print()
        print("=" * 55)
        print("DATASET SAVED")
        print("=" * 55)
        print("Data shape:", data_array.shape)
        print("Labels shape:", labels_array.shape)
        print("Total sequences:", len(data_array))

    else:

        print()
        print("No sequences were collected.")

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print()
    print("Data collection finished.")