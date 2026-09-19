import os
import json
import random
import warnings

import numpy as np
import librosa
import soundfile as sf

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"combined"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

OUTPUT_DIR = "models"

MODEL_H5 = os.path.join(
    OUTPUT_DIR,
    "baby_audio_strong.keras"
)

MODEL_TFLITE = os.path.join(
    OUTPUT_DIR,
    "baby_audio_strong.tflite"
)

LABEL_MAP_PATH = os.path.join(
    OUTPUT_DIR,
    "label_map.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 24

WIN_LENGTH = 0.025

HOP_LENGTH = 0.010

N_FFT = 1024

MAX_FRAMES = 500

MIN_AUDIO_LENGTH = 0.10


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.0005

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# GPU CONFIGURATION
# ============================================================

print("=" * 70)
print("BABY AUDIO CLASSIFICATION - STRONG TRAINING")
print("=" * 70)

print("\nTensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
else:
    print("GPU not detected - using CPU")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DISCOVER CLASSES
# ============================================================

def get_classes():

    if not os.path.isdir(TRAIN_DIR):
        raise RuntimeError(
            f"Training directory not found:\n{TRAIN_DIR}"
        )

    classes = []

    for name in os.listdir(TRAIN_DIR):

        path = os.path.join(TRAIN_DIR, name)

        if os.path.isdir(path):
            classes.append(name)

    classes = sorted(classes)

    if len(classes) == 0:
        raise RuntimeError(
            "No class folders found in train directory."
        )

    return classes


CLASSES = get_classes()

NUM_CLASSES = len(CLASSES)

print("\nDetected classes:")

for i, cls in enumerate(CLASSES):
    print(f"{i}: {cls}")

print("\nNumber of classes:", NUM_CLASSES)


# ============================================================
# SAVE LABEL MAP
# ============================================================

label_map = {
    str(i): cls
    for i, cls in enumerate(CLASSES)
}

with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(
        label_map,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# AUDIO FILE EXTENSIONS
# ============================================================

AUDIO_EXTENSIONS = (
    ".wav",
    ".wave",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
)


# ============================================================
# FIND AUDIO FILES
# ============================================================

def find_audio_files(directory):

    files = []

    if not os.path.isdir(directory):
        print(
            f"[WARNING] Directory does not exist: {directory}"
        )
        return files

    for class_index, class_name in enumerate(CLASSES):

        class_dir = os.path.join(
            directory,
            class_name
        )

        if not os.path.isdir(class_dir):

            print(
                f"[WARNING] Missing class folder: "
                f"{class_dir}"
            )

            continue

        for root, _, filenames in os.walk(class_dir):

            for filename in filenames:

                if filename.lower().endswith(
                    AUDIO_EXTENSIONS
                ):

                    full_path = os.path.join(
                        root,
                        filename
                    )

                    files.append(
                        (
                            full_path,
                            class_index
                        )
                    )

    return files


# ============================================================
# DATASET FILES
# ============================================================

train_files = find_audio_files(TRAIN_DIR)

val_files = find_audio_files(VAL_DIR)

test_files = find_audio_files(TEST_DIR)


print("\nDataset files:")

print("Training   :", len(train_files))

print("Validation :", len(val_files))

print("Test       :", len(test_files))


if len(train_files) == 0:
    raise RuntimeError(
        "No training audio files found."
    )

if len(val_files) == 0:
    raise RuntimeError(
        "No validation audio files found."
    )

if len(test_files) == 0:
    raise RuntimeError(
        "No test audio files found."
    )


# ============================================================
# MFCC EXTRACTION
#
# IMPORTANT:
# This intentionally matches your old Raspberry Pi code.
#
# Old Pi:
#
# mfcc(
#     signal,
#     samplerate=sr,
#     numcep=24,
#     winlen=0.025,
#     winstep=0.01,
#     nfft=1024
# )
#
# Here we reproduce the same basic parameters with librosa.
# ============================================================

def extract_mfcc(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if audio is None or len(audio) == 0:
            raise ValueError(
                "Empty audio"
            )

        audio = audio.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Remove DC offset
        # ----------------------------------------------------

        audio = audio - np.mean(audio)

        # ----------------------------------------------------
        # Normalize safely
        # ----------------------------------------------------

        peak = np.max(
            np.abs(audio)
        )

        if peak > 1e-8:

            audio = audio / peak

        # ----------------------------------------------------
        # MFCC
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=int(
                SAMPLE_RATE * HOP_LENGTH
            ),
            win_length=int(
                SAMPLE_RATE * WIN_LENGTH
            ),
            window="hann",
            center=True
        )

        # librosa returns:
        #
        # 24 x frames
        #
        # We need:
        #
        # frames x 24

        mfcc = mfcc.T

        # ----------------------------------------------------
        # Fixed length
        # ----------------------------------------------------

        if mfcc.shape[0] < MAX_FRAMES:

            pad_amount = (
                MAX_FRAMES -
                mfcc.shape[0]
            )

            mfcc = np.pad(
                mfcc,
                (
                    (0, pad_amount),
                    (0, 0)
                ),
                mode="constant"
            )

        else:

            mfcc = mfcc[
                :MAX_FRAMES,
                :
            ]

        # ----------------------------------------------------
        # Per-feature standardization
        #
        # This is performed later using training statistics.
        # ----------------------------------------------------

        return mfcc.astype(
            np.float32
        )

    except Exception as e:

        print(
            f"[AUDIO ERROR] {file_path}"
        )

        print(
            "Reason:",
            str(e)
        )

        return None


# ============================================================
# LOAD FEATURE DATASET
# ============================================================

def load_dataset(file_list):

    X = []

    y = []

    bad_files = []

    total = len(file_list)

    for index, (path, label) in enumerate(
        file_list
    ):

        if index % 100 == 0:

            print(
                f"Processing "
                f"{index}/{total}"
            )

        features = extract_mfcc(
            path
        )

        if features is None:

            bad_files.append(path)

            continue

        X.append(features)

        y.append(label)

    if len(X) == 0:

        raise RuntimeError(
            "No valid audio features."
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    return X, y, bad_files


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING TRAIN FEATURES")
print("=" * 70)

X_train, y_train, bad_train = load_dataset(
    train_files
)


print("\n" + "=" * 70)
print("EXTRACTING VALIDATION FEATURES")
print("=" * 70)

X_val, y_val, bad_val = load_dataset(
    val_files
)


print("\n" + "=" * 70)
print("EXTRACTING TEST FEATURES")
print("=" * 70)

X_test, y_test, bad_test = load_dataset(
    test_files
)


print("\nFeature shapes:")

print("X_train:", X_train.shape)

print("X_val  :", X_val.shape)

print("X_test :", X_test.shape)


# ============================================================
# FEATURE STANDARDIZATION
#
# IMPORTANT:
# Statistics are calculated ONLY from training data.
#
# This prevents validation/test leakage.
# ============================================================

mean = np.mean(
    X_train,
    axis=(0, 1),
    keepdims=True
)

std = np.std(
    X_train,
    axis=(0, 1),
    keepdims=True
)

std = np.maximum(
    std,
    1e-6
)


X_train = (
    X_train - mean
) / std


X_val = (
    X_val - mean
) / std


X_test = (
    X_test - mean
) / std


# ============================================================
# SAVE NORMALIZATION PARAMETERS
# ============================================================

normalization_data = {

    "mean": mean.reshape(-1).tolist(),

    "std": std.reshape(-1).tolist(),

    "sample_rate": SAMPLE_RATE,

    "n_mfcc": N_MFCC,

    "n_fft": N_FFT,

    "win_length": WIN_LENGTH,

    "hop_length": HOP_LENGTH,

    "max_frames": MAX_FRAMES
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "normalization.json"
    ),
    "w"
) as f:

    json.dump(
        normalization_data,
        f,
        indent=4
    )


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

X_train = np.expand_dims(
    X_train,
    axis=-1
)

X_val = np.expand_dims(
    X_val,
    axis=-1
)

X_test = np.expand_dims(
    X_test,
    axis=-1
)


print("\nFinal model input:")

print(
    X_train.shape
)


# ============================================================
# CLASS WEIGHTS
#
# Helps if your 5 classes don't contain exactly the same
# number of training samples.
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for i, weight in class_weights.items():

    print(
        CLASSES[i],
        ":",
        weight
    )


# ============================================================
# STRONG CNN MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(
            MAX_FRAMES,
            N_MFCC,
            1
        )
    )

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(
        0.15
    )(x)


    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(
        0.20
    )(x)


    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(
        0.25
    )(x)


    # --------------------------------------------------------
    # Global feature extraction
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)


    # --------------------------------------------------------
    # Dense classifier
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(
            1e-4
        )
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.35
    )(x)


    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)


    model = models.Model(
        inputs,
        outputs
    )

    return model


model = build_model()


# ============================================================
# COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\n" + "=" * 70)

model.summary()

print("=" * 70)


# ============================================================
# CALLBACKS
# ============================================================

early_stop = callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=15,

    mode="max",

    restore_best_weights=True,

    verbose=1
)


reduce_lr = callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-7,

    verbose=1
)


checkpoint = callbacks.ModelCheckpoint(

    MODEL_H5,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ],

    shuffle=True,

    verbose=1
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_json = {

    key: [
        float(v)
        for v in values
    ]

    for key, values
    in history.history.items()
}


with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history_json,
        f,
        indent=4
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_H5
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)


val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)


print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# TEST ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"\nExact test accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    predictions,

    target_names=CLASSES,

    digits=4
)


print("\nClassification report:\n")

print(report)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


plt.figure(
    figsize=(10, 8)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=CLASSES,

    yticklabels=CLASSES

)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Baby Audio Classification - Test Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200
)

plt.close()


# ============================================================
# SAVE TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Training Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_curve.png"
    ),
    dpi=200
)

plt.close()


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# TFLITE CONVERSION
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING TO TFLITE")
print("=" * 70)


converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)


# ------------------------------------------------------------
# Float32 model
#
# This is safest for your first Pi deployment.
# ------------------------------------------------------------

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]


tflite_model = converter.convert()


with open(
    MODEL_TFLITE,
    "wb"
) as f:

    f.write(
        tflite_model
    )


print(
    "TFLite model saved:"
)

print(
    MODEL_TFLITE
)


# ============================================================
# TFLITE MODEL TEST
# ============================================================

print("\n" + "=" * 70)
print("TESTING TFLITE MODEL")
print("=" * 70)


interpreter = tf.lite.Interpreter(
    model_path=MODEL_TFLITE
)

interpreter.allocate_tensors()


input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)


print(
    "TFLite input shape:",
    input_details[0]["shape"]
)

print(
    "TFLite input type:",
    input_details[0]["dtype"]
)

print(
    "TFLite output shape:",
    output_details[0]["shape"]
)


# ------------------------------------------------------------
# Compare TFLite predictions
# ------------------------------------------------------------

tflite_predictions = []


for i in range(
    len(X_test)
):

    sample = X_test[
        i:i + 1
    ]


    # Ensure exact input dtype

    sample = sample.astype(
        input_details[0]["dtype"]
    )


    interpreter.set_tensor(
        input_details[0]["index"],
        sample
    )


    interpreter.invoke()


    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]


    predicted = np.argmax(
        output
    )


    tflite_predictions.append(
        predicted
    )


tflite_predictions = np.asarray(
    tflite_predictions
)


tflite_accuracy = accuracy_score(
    y_test,
    tflite_predictions
)


print(
    f"TFLite test accuracy: "
    f"{tflite_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Classes          : {NUM_CLASSES}"
)

print(
    f"Train samples    : {len(X_train)}"
)

print(
    f"Validation       : {len(X_val)}"
)

print(
    f"Test samples     : {len(X_test)}"
)

print(
    f"Keras accuracy   : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"TFLite accuracy  : "
    f"{tflite_accuracy * 100:.2f}%"
)

print(
    "\nModel files:"
)

print(
    MODEL_H5
)

print(
    MODEL_TFLITE
)

print(
    LABEL_MAP_PATH
)

print(
    os.path.join(
        OUTPUT_DIR,
        "normalization.json"
    )
)

print(
    CONFUSION_MATRIX_PATH
)

print(
    REPORT_PATH
)

print("=" * 70)