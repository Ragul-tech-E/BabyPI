import os
import json
import time

import numpy as np
import librosa
from ai_edge_litert import interpreter as tflite


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

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
# CHECK MODEL FILES
# ============================================================

print("=" * 70)
print("BABY AUDIO - RASPBERRY PI ZERO 2 W")
print("=" * 70)

print("\nChecking model files...")

if not os.path.exists(MODEL_TFLITE):
    raise FileNotFoundError(
        f"\nTFLite model not found:\n{MODEL_TFLITE}"
    )

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(
        f"\nLabel map not found:\n{LABEL_MAP_PATH}"
    )

if not os.path.exists(NORMALIZATION_PATH):
    raise FileNotFoundError(
        f"\nNormalization file not found:\n{NORMALIZATION_PATH}"
    )

print("Model files found.")


# ============================================================
# LOAD LABEL MAP
# ============================================================

with open(
    LABEL_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:

    label_map = json.load(f)


CLASSES = {
    int(k): v
    for k, v in label_map.items()
}


print("\nClasses:")

for index in sorted(CLASSES):
    print(f"  {index}: {CLASSES[index]}")


# ============================================================
# LOAD NORMALIZATION PARAMETERS
# ============================================================

with open(
    NORMALIZATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    normalization = json.load(f)


# ============================================================
# AUDIO PARAMETERS
# ============================================================

SAMPLE_RATE = int(
    normalization["sample_rate"]
)

N_MFCC = int(
    normalization["n_mfcc"]
)

N_FFT = int(
    normalization["n_fft"]
)

WIN_LENGTH = float(
    normalization["win_length"]
)

HOP_LENGTH = float(
    normalization["hop_length"]
)

MAX_FRAMES = int(
    normalization["max_frames"]
)


# ============================================================
# NORMALIZATION
# ============================================================

MEAN = np.asarray(
    normalization["mean"],
    dtype=np.float32
).reshape(
    1,
    N_MFCC
)

STD = np.asarray(
    normalization["std"],
    dtype=np.float32
).reshape(
    1,
    N_MFCC
)

STD = np.maximum(
    STD,
    1e-6
)


# ============================================================
# PRINT PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("AUDIO / MFCC PARAMETERS")
print("=" * 70)

print(f"Sample rate : {SAMPLE_RATE} Hz")
print(f"MFCC        : {N_MFCC}")
print(f"N_FFT       : {N_FFT}")
print(f"Win length  : {WIN_LENGTH}")
print(f"Hop length  : {HOP_LENGTH}")
print(f"Max frames  : {MAX_FRAMES}")


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TFLITE MODEL")
print("=" * 70)

load_start = time.perf_counter()

interpreter = tflite.Interpreter(
    model_path=MODEL_TFLITE,
    num_threads=4
)

interpreter.allocate_tensors()

load_time = time.perf_counter() - load_start

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ============================================================
# MODEL INFORMATION
# ============================================================

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

print(
    f"Model load time: {load_time:.3f} seconds"
)


# ============================================================
# MFCC EXTRACTION
# ============================================================

def extract_mfcc(file_path):

    print("\n" + "=" * 70)
    print("AUDIO PROCESSING")
    print("=" * 70)

    print("\nAudio file:")
    print(file_path)

    processing_start = time.perf_counter()

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    print("\nLoading audio...")

    audio, original_sr = librosa.load(
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
    # REMOVE DC OFFSET
    # --------------------------------------------------------

    audio = audio - np.mean(audio)

    # --------------------------------------------------------
    # PEAK NORMALIZATION
    # --------------------------------------------------------

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    print("Extracting MFCC...")

    mfcc_start = time.perf_counter()

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

    mfcc_time = (
        time.perf_counter()
        - mfcc_start
    )

    # --------------------------------------------------------
    # TRANSPOSE
    #
    # 24 x frames
    #
    # becomes
    #
    # frames x 24
    # --------------------------------------------------------

    mfcc = mfcc.T

    original_frames = mfcc.shape[0]

    # --------------------------------------------------------
    # FIX FRAME COUNT
    # --------------------------------------------------------

    if original_frames < MAX_FRAMES:

        pad_amount = (
            MAX_FRAMES
            - original_frames
        )

        print(
            f"MFCC frames: {original_frames}"
        )

        print(
            f"Padding: {pad_amount} frames"
        )

        mfcc = np.pad(

            mfcc,

            (
                (0, pad_amount),
                (0, 0)
            ),

            mode="constant"
        )

        frame_action = (
            f"PAD {pad_amount} frames"
        )

    else:

        if original_frames > MAX_FRAMES:

            print(
                f"MFCC frames: {original_frames}"
            )

            print(
                f"Truncating to {MAX_FRAMES} frames"
            )

        mfcc = mfcc[
            :MAX_FRAMES,
            :
        ]

        frame_action = (
            f"TRUNCATE to {MAX_FRAMES}"
        )

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    mfcc = (
        mfcc - MEAN
    ) / STD

    # --------------------------------------------------------
    # CHANNEL DIMENSION
    #
    # frames x MFCC
    #
    # ->
    #
    # frames x MFCC x 1
    # --------------------------------------------------------

    mfcc = np.expand_dims(
        mfcc,
        axis=-1
    )

    # --------------------------------------------------------
    # BATCH DIMENSION
    #
    # ->
    #
    # 1 x frames x MFCC x 1
    # --------------------------------------------------------

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    mfcc = mfcc.astype(
        np.float32
    )

    processing_time = (
        time.perf_counter()
        - processing_start
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print("\nAudio information:")

    print(
        f"Original sample rate : "
        f"{original_sr} Hz"
    )

    print(
        f"Model sample rate    : "
        f"{SAMPLE_RATE} Hz"
    )

    print(
        f"Audio samples        : "
        f"{len(audio)}"
    )

    print(
        f"Audio duration       : "
        f"{len(audio) / SAMPLE_RATE:.2f} sec"
    )

    print(
        f"Original MFCC frames : "
        f"{original_frames}"
    )

    print(
        f"Frame action         : "
        f"{frame_action}"
    )

    print(
        f"Final feature shape  : "
        f"{mfcc.shape}"
    )

    print(
        f"MFCC processing time : "
        f"{processing_time:.3f} sec"
    )

    print(
        f"MFCC calculation     : "
        f"{mfcc_time:.3f} sec"
    )

    return mfcc


# ============================================================
# GET AUDIO PATH
# ============================================================

print("\n" + "=" * 70)
print("ENTER AUDIO FILE")
print("=" * 70)

print(
    "\nExample:"
)

print(
    "/home/pi/baby_model/audio/test.wav"
)

print(
    "or"
)

print(
    "/home/pi/test.wav"
)

audio_path = input(
    "\nEnter audio path: "
).strip()


# ============================================================
# REMOVE QUOTES
# ============================================================

audio_path = audio_path.strip(
    '"'
).strip(
    "'"
)


# ============================================================
# EXPAND ~
# ============================================================

audio_path = os.path.expanduser(
    audio_path
)


# ============================================================
# CONVERT TO ABSOLUTE PATH
# ============================================================

audio_path = os.path.abspath(
    audio_path
)


# ============================================================
# CHECK AUDIO
# ============================================================

if not os.path.isfile(audio_path):

    raise FileNotFoundError(
        f"\nAudio file not found:\n"
        f"{audio_path}"
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

features = extract_mfcc(
    audio_path
)


# ============================================================
# CHECK MODEL INPUT
# ============================================================

expected_shape = tuple(
    input_details[0]["shape"]
)

actual_shape = tuple(
    features.shape
)

print("\n" + "=" * 70)
print("MODEL INPUT CHECK")
print("=" * 70)

print(
    "Expected:",
    expected_shape
)

print(
    "Actual  :",
    actual_shape
)


# ============================================================
# HANDLE DYNAMIC INPUT SHAPE
# ============================================================

if list(expected_shape) != list(actual_shape):

    # Some TFLite models may contain
    # dynamic dimensions.

    try:

        interpreter.resize_tensor_input(
            input_details[0]["index"],
            actual_shape,
            strict=False
        )

        interpreter.allocate_tensors()

        input_details = (
            interpreter.get_input_details()
        )

        output_details = (
            interpreter.get_output_details()
        )

        expected_shape = tuple(
            input_details[0]["shape"]
        )

        print(
            "Input tensor resized successfully."
        )

        print(
            "New input shape:",
            expected_shape
        )

    except Exception as e:

        raise ValueError(
            "\nMODEL INPUT SHAPE MISMATCH\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {actual_shape}\n"
            f"Resize failed: {e}"
        )


# ============================================================
# MATCH INPUT DTYPE
# ============================================================

input_dtype = (
    input_details[0]["dtype"]
)

features = features.astype(
    input_dtype
)


# ============================================================
# RUN INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING INFERENCE")
print("=" * 70)

inference_start = time.perf_counter()

interpreter.set_tensor(
    input_details[0]["index"],
    features
)

interpreter.invoke()

inference_time = (
    time.perf_counter()
    - inference_start
)


# ============================================================
# GET OUTPUT
# ============================================================

output = interpreter.get_tensor(
    output_details[0]["index"]
)

output = output[0]


# ============================================================
# DEQUANTIZE OUTPUT
# ============================================================

output_quantization = (
    output_details[0].get(
        "quantization",
        (0.0, 0)
    )
)

scale, zero_point = (
    output_quantization
)

if (
    output_details[0]["dtype"] != np.float32
    and scale != 0
):

    output = (
        output.astype(np.float32)
        - zero_point
    ) * scale


# ============================================================
# CONVERT OUTPUT TO PROBABILITIES
# ============================================================

output = np.asarray(
    output,
    dtype=np.float32
)

output_sum = np.sum(
    output
)

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
    np.argmax(
        probabilities
    )
)

predicted_class = CLASSES.get(
    predicted_index,
    f"Unknown class {predicted_index}"
)

confidence = float(
    probabilities[
        predicted_index
    ]
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    "\nPredicted class:"
)

print(
    f"  {predicted_class}"
)

print(
    "\nConfidence:"
)

print(
    f"  {confidence * 100:.2f}%"
)

print(
    "\nInference time:"
)

print(
    f"  {inference_time:.3f} seconds"
)


# ============================================================
# ALL CLASS PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("CLASS PROBABILITIES")
print("=" * 70)

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


# ============================================================
# SORT
# ============================================================

results.sort(
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# PRINT
# ============================================================

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
        "Very low confidence - "
        "prediction may be unreliable."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print(
    f"\nAudio: {audio_path}"
)

print(
    f"Prediction: {predicted_class}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

print(
    f"Inference: {inference_time:.3f} sec"
)

print(
    "\nDone."
)
