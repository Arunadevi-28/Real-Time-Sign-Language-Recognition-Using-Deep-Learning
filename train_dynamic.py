import os
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "dynamic_data.npy"
LABEL_FILE = "dynamic_labels.npy"

MODEL_FILE = "dynamic_model.keras"
LABEL_JSON = "dynamic_labels.json"

SEQUENCE_LENGTH = 30
FEATURES = 63

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

NUM_CLASSES = len(LABELS)

# Minimum sequences required for each class
MINIMUM_PER_CLASS = 20

# ============================================================
# HEADER
# ============================================================

print()
print("=" * 55)
print("          DYNAMIC LSTM MODEL TRAINING")
print("=" * 55)
print()

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(DATA_FILE):

    print("ERROR:", DATA_FILE, "not found.")
    print("Run:")
    print("python collect_dynamic.py")
    exit()

if not os.path.exists(LABEL_FILE):

    print("ERROR:", LABEL_FILE, "not found.")
    print("Run:")
    print("python collect_dynamic.py")
    exit()

# ============================================================
# LOAD DATA
# ============================================================

X = np.load(DATA_FILE)
y = np.load(LABEL_FILE)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print()

# ============================================================
# SHAPE CHECK
# ============================================================

if len(X.shape) != 3:

    print("ERROR: Dataset must have 3 dimensions.")
    print("Expected:")
    print("(samples, 30, 63)")
    print("Got:", X.shape)
    exit()

if X.shape[1] != SEQUENCE_LENGTH:

    print("ERROR: Sequence length must be 30.")
    print("Got:", X.shape[1])
    exit()

if X.shape[2] != FEATURES:

    print("ERROR: Feature count must be 63.")
    print("Got:", X.shape[2])
    print()
    print("Delete the old dynamic dataset and recollect it.")
    exit()

if len(X) != len(y):

    print("ERROR: X and y lengths do not match.")
    exit()

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("Class distribution")
print("-" * 45)

class_counts = {}

for class_id, label in enumerate(LABELS):

    count = int(np.sum(y == class_id))

    class_counts[label] = count

    print(
        f"{label:<12}: {count}"
    )

print("-" * 45)
print()

# ============================================================
# MINIMUM DATA CHECK
# ============================================================

for label, count in class_counts.items():

    if count < MINIMUM_PER_CLASS:

        print(
            f"ERROR: {label} has only {count} sequences."
        )

        print(
            f"Collect at least {MINIMUM_PER_CLASS} "
            "sequences for every sign."
        )

        print()
        print(
            "Recommended: 30 sequences per sign."
        )

        exit()

# ============================================================
# CONVERT TYPE
# ============================================================

X = X.astype(np.float32)
y = y.astype(np.int32)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print()

# ============================================================
# MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, FEATURES)
    ),

    tf.keras.layers.Masking(
        mask_value=0.0
    ),

    tf.keras.layers.LSTM(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.LSTM(
        64
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print()
print("Model:")
model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=0.00001
)

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 55)
print("STARTING TRAINING")
print("=" * 55)
print()

history = model.fit(

    X_train,
    y_train,

    validation_split=0.20,

    epochs=60,

    batch_size=16,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

# ============================================================
# TEST
# ============================================================

print()
print("=" * 55)
print("TESTING MODEL")
print("=" * 55)
print()

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)

# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABELS,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print(cm)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_FILE)

# ============================================================
# SAVE LABELS
# ============================================================

with open(
    LABEL_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        LABELS,
        file,
        indent=4
    )

# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)

print()
print("Created:")
print("1.", MODEL_FILE)
print("2.", LABEL_JSON)

print()
print("You can now run:")
print("python realtime_dynamic.py")
print()