import os
import json
import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# MODEL FILES
# ============================================================

MODEL_DIR = "Librosa_models"

MODEL_TFLITE = os.path.join(
    MODEL_DIR,
    "baby_audio_strong.tflite"
)

LABEL_MAP_PATH = os.path.join(
    MODEL_DIR,
    "label_map.json"
)

NORMALIZATION_PATH = os.path.join(
    MODEL_DIR,
    "normalization.json"
)


# ============================================================
# LOAD LABEL MAP
# ============================================================

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(
        f"Label map not found:\n{LABEL_MAP_PATH}"
    )

with open(
    LABEL_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:

    label_map = json.load(f)


# Convert JSON keys to integers

CLASSES = {
    int(k): v
    for k, v in label_map.items()
}


# ============================================================
# LOAD NORMALIZATION PARAMETERS
# ============================================================

if not os.path.exists(NORMALIZATION_PATH):
    raise FileNotFoundError(
        f"Normalization file not found:\n"
        f"{NORMALIZATION_PATH}"
    )

with open(
    NORMALIZATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    normalization = json.load(f)


# ============================================================
# AUDIO PARAMETERS
#
# These MUST match training.
# ============================================================

SAMPLE_RATE = normalization["sample_rate"]

N_MFCC = normalization["n_mfcc"]

N_FFT = normalization["n_fft"]

WIN_LENGTH = normalization["win_length"]

HOP_LENGTH = normalization["hop_length"]

MAX_FRAMES = normalization["max_frames"]


# ============================================================
# NORMALIZATION ARRAYS
# ============================================================

MEAN = np.asarray(
    normalization["mean"],
    dtype=np.float32
).reshape(1, N_MFCC)

STD = np.asarray(
    normalization["std"],
    dtype=np.float32
).reshape(1, N_MFCC)

STD = np.maximum(
    STD,
    1e-6
)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

if not os.path.exists(MODEL_TFLITE):

    raise FileNotFoundError(
        f"TFLite model not found:\n"
        f"{MODEL_TFLITE}"
    )


print("=" * 70)
print("BABY AUDIO TFLITE TEST")
print("=" * 70)

print("\nLoading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_TFLITE
)

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()


print("\nModel loaded successfully.")

print(
    "Input shape :",
    input_details[0]["shape"]
)

print(
    "Input dtype :",
    input_details[0]["dtype"]
)

print(
    "Output shape:",
    output_details[0]["shape"]
)

print(
    "Output dtype:",
    output_details[0]["dtype"]
)


# ============================================================
# AUDIO PREPROCESSING
#
# EXACTLY MATCHES TRAINING CODE
# ============================================================

def extract_mfcc(file_path):

    print("\nLoading audio:")
    print(file_path)

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio is None or len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )


    audio = audio.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    audio = audio - np.mean(audio)


    # --------------------------------------------------------
    # Peak normalization
    # --------------------------------------------------------

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak


    # --------------------------------------------------------
    # MFCC
    #
    # SAME PARAMETERS AS TRAINING
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Convert:
    #
    # 24 x frames
    #
    # to:
    #
    # frames x 24
    # --------------------------------------------------------

    mfcc = mfcc.T


    # --------------------------------------------------------
    # Fixed 500 frames
    #
    # SAME AS TRAINING
    # --------------------------------------------------------

    original_frames = mfcc.shape[0]

    if original_frames < MAX_FRAMES:

        pad_amount = (
            MAX_FRAMES -
            original_frames
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


    # --------------------------------------------------------
    # Feature standardization
    #
    # Use training mean/std.
    # DO NOT calculate mean/std from test audio.
    # --------------------------------------------------------

    mfcc = (
        mfcc - MEAN
    ) / STD


    # --------------------------------------------------------
    # Add channel dimension
    #
    # frames x MFCC
    #
    # becomes
    #
    # frames x MFCC x 1
    # --------------------------------------------------------

    mfcc = np.expand_dims(
        mfcc,
        axis=-1
    )


    # --------------------------------------------------------
    # Add batch dimension
    #
    # becomes:
    #
    # 1 x 500 x 24 x 1
    # --------------------------------------------------------

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )


    mfcc = mfcc.astype(
        np.float32
    )


    print("\nAudio information:")

    print(
        "Original sample rate:",
        sr
    )

    print(
        "Audio samples:",
        len(audio)
    )

    print(
        "Audio duration:",
        f"{len(audio) / SAMPLE_RATE:.2f} seconds"
    )

    print(
        "Original MFCC frames:",
        original_frames
    )

    print(
        "Final feature shape:",
        mfcc.shape
    )


    return mfcc


# ============================================================
# GET TEST AUDIO PATH
# ============================================================

print("\n" + "=" * 70)
print("ENTER TEST AUDIO")
print("=" * 70)

print(
    "\nExample:"
)

print(
    r"C:\Users\YourName\Desktop\test.wav"
)

audio_path = input(
    "\nEnter full audio file path: "
).strip()


# Remove quotes if user pasted:
#
# "C:\audio\test.wav"
#
# or
#
# 'C:\audio\test.wav'

audio_path = audio_path.strip(
    '"'
).strip(
    "'"
)


# ============================================================
# CHECK AUDIO FILE
# ============================================================

if not os.path.isfile(audio_path):

    raise FileNotFoundError(
        f"\nAudio file not found:\n{audio_path}"
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

features = extract_mfcc(
    audio_path
)


# ============================================================
# CHECK MODEL INPUT SHAPE
# ============================================================

expected_shape = (
    input_details[0]["shape"]
)

print(
    "\nExpected model input:",
    expected_shape
)

print(
    "Actual input:",
    features.shape
)


# ============================================================
# MATCH TFLITE INPUT DTYPE
# ============================================================

input_dtype = input_details[0]["dtype"]

features = features.astype(
    input_dtype
)


# ============================================================
# RUN INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING INFERENCE")
print("=" * 70)


interpreter.set_tensor(

    input_details[0]["index"],

    features
)


interpreter.invoke()


# ============================================================
# GET OUTPUT
# ============================================================

output = interpreter.get_tensor(
    output_details[0]["index"]
)


output = output[0]


# ============================================================
# HANDLE QUANTIZED OUTPUT IF NECESSARY
# ============================================================

output_quantization = (
    output_details[0].get(
        "quantization",
        (0.0, 0)
    )
)

scale, zero_point = output_quantization


if (
    output_details[0]["dtype"] != np.float32
    and scale != 0
):

    output = (
        output.astype(np.float32)
        - zero_point
    ) * scale


# ============================================================
# NORMALIZE OUTPUT IF NEEDED
# ============================================================

output_sum = np.sum(output)

if output_sum > 0:

    probabilities = (
        output / output_sum
    )

else:

    probabilities = output


# ============================================================
# PREDICTION
# ============================================================

predicted_index = int(
    np.argmax(probabilities)
)


predicted_class = CLASSES.get(
    predicted_index,
    f"Unknown class {predicted_index}"
)


confidence = float(
    probabilities[predicted_index]
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)


print(
    "\nPredicted class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)


# ============================================================
# ALL CLASS PROBABILITIES
# ============================================================

print("\nClass probabilities:")

results = []

for index, probability in enumerate(
    probabilities
):

    class_name = CLASSES.get(
        index,
        f"Class {index}"
    )

    results.append(
        (
            class_name,
            float(probability)
        )
    )


# Sort highest to lowest

results.sort(
    key=lambda x: x[1],
    reverse=True
)


for rank, (
    class_name,
    probability
) in enumerate(
    results,
    start=1
):

    print(
        f"{rank}. "
        f"{class_name:<25} "
        f"{probability * 100:>7.2f}%"
    )


# ============================================================
# CONFIDENCE INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)


if confidence >= 0.90:

    print(
        "Very high confidence"
    )

elif confidence >= 0.75:

    print(
        "High confidence"
    )

elif confidence >= 0.60:

    print(
        "Moderate confidence"
    )

elif confidence >= 0.40:

    print(
        "Low confidence"
    )

else:

    print(
        "Very low confidence - prediction may be unreliable."
    )


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)