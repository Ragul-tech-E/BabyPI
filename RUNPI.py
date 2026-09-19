# ============================================================
# PI ZERO 2 W REALISTIC TFLITE TESTER
#
# PURPOSE
# -------
# Test the trained baby_audio_strong.tflite model on a PC
# while deliberately using a Raspberry Pi Zero 2 W-like
# inference configuration.
#
# IMPORTANT:
# The PC CPU is NOT the same as the Pi CPU.
# Therefore the measured milliseconds are NOT an exact Pi
# timing prediction.
#
# What IS matched:
#   - Same TFLite model
#   - CPU inference
#   - Batch size = 1
#   - 1 / 2 / 4 TFLite threads
#   - Same MFCC implementation
#   - Same MFCC parameters
#   - Same resampling
#   - Same 500-frame handling
#   - Same input shape
#
#
# AUDIO PIPELINE
# --------------
#
# WAV
#  |
#  +--> soundfile
#  |
#  +--> stereo -> mono
#  |
#  +--> resample -> 16000 Hz
#  |
#  +--> python_speech_features.mfcc
#  |
#  +--> 24 MFCC
#  |
#  +--> 500 frames
#  |
#  +--> [1, 500, 24, 1]
#  |
#  +--> TFLite CPU
#  |
#  +--> prediction
#
#
# PI ZERO 2 W
# -----------
# Quad-core ARM Cortex-A53
#
# This program benchmarks:
#
#   1 TFLite thread
#   2 TFLite threads
#   4 TFLite threads
#
# so you can choose the best deployment setting.
#
# ============================================================


# ============================================================
# 1. ENVIRONMENT
# ============================================================

import os

# NEVER use the PC GPU for this test.
# This prevents TensorFlow from using your GTX 1660 Super.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Keep TensorFlow quiet if it is used as PC fallback.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Do not enable deterministic GPU operations.
os.environ["TF_DETERMINISTIC_OPS"] = "0"


# ============================================================
# 2. IMPORTS
# ============================================================

import sys
import time
import json
import gc
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

from scipy.signal import resample

from python_speech_features import mfcc


# ============================================================
# 3. OPTIONAL SYSTEM MONITOR
# ============================================================

try:
    import psutil

    PSUTIL_AVAILABLE = True

except ImportError:

    PSUTIL_AVAILABLE = False


# ============================================================
# 4. SETTINGS
# ============================================================

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_PATH = Path(
    "models"
) / "baby_audio_strong.tflite"


LABEL_MAP_PATH = Path(
    "models"
) / "label_map.json"


# ------------------------------------------------------------
# EXACT AUDIO / MFCC SETTINGS
# ------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000

NUM_MFCC = 24

WINLEN = 0.025

WINSTEP = 0.01

NFFT = 1024

MAX_FRAMES = 500


# ------------------------------------------------------------
# PI ZERO 2 W SETTINGS
# ------------------------------------------------------------

# Pi Zero 2 W has 4 CPU cores.
#
# We benchmark all three practical configurations.
THREAD_MODES = [1, 2, 4]


# ------------------------------------------------------------
# Number of repeated inference measurements
# ------------------------------------------------------------

# Higher = more stable timing.
#
# 20 is enough for normal testing.
# Increase to 50 if you want more accurate timing.
BENCHMARK_RUNS = 20


# ------------------------------------------------------------
# Number of warm-up runs
# ------------------------------------------------------------

WARMUP_RUNS = 3


# ============================================================
# 5. GLOBAL MODEL CLASS
# ============================================================

class TFLiteRunner:

    def __init__(
        self,
        model_path,
        num_threads
    ):

        self.model_path = str(
            model_path
        )

        self.num_threads = int(
            num_threads
        )

        self.interpreter = None

        self.input_details = None

        self.output_details = None

        self.input_index = None

        self.output_index = None

        self.load()


    # ========================================================
    # LOAD INTERPRETER
    # ========================================================

    def load(self):

        print(
            f"\nLoading TFLite "
            f"with {self.num_threads} thread(s)..."
        )

        # ----------------------------------------------------
        # First choice:
        # tflite_runtime
        #
        # This is what we want on Raspberry Pi.
        # ----------------------------------------------------

        try:

            from tflite_runtime.interpreter import (
                Interpreter
            )

            print(
                "Runtime: tflite_runtime"
            )

            self.interpreter = Interpreter(
                model_path=self.model_path,
                num_threads=self.num_threads
            )

        except ImportError:

            # ------------------------------------------------
            # PC fallback:
            # TensorFlow Lite
            # ------------------------------------------------

            try:

                import tensorflow as tf

                print(
                    "Runtime: TensorFlow Lite"
                )

                self.interpreter = (
                    tf.lite.Interpreter(
                        model_path=self.model_path,
                        num_threads=self.num_threads
                    )
                )

            except ImportError:

                raise RuntimeError(
                    "\nNeither tflite_runtime nor "
                    "TensorFlow is installed.\n\n"
                    "Install TensorFlow 2.10.1:\n"
                    "pip install tensorflow==2.10.1"
                )


        # ----------------------------------------------------
        # Allocate tensors
        # ----------------------------------------------------

        self.interpreter.allocate_tensors()


        # ----------------------------------------------------
        # Get input/output
        # ----------------------------------------------------

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        self.input_index = (
            self.input_details[0]["index"]
        )

        self.output_index = (
            self.output_details[0]["index"]
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def print_info(self):

        print()
        print("-" * 70)

        print(
            f"TFLite threads : {self.num_threads}"
        )

        print(
            "Input shape    :",
            self.input_details[0]["shape"]
        )

        if "shape_signature" in self.input_details[0]:

            print(
                "Input signature:",
                self.input_details[0][
                    "shape_signature"
                ]
            )

        print(
            "Input dtype    :",
            self.input_details[0]["dtype"]
        )

        print(
            "Output shape   :",
            self.output_details[0]["shape"]
        )

        print(
            "Output dtype   :",
            self.output_details[0]["dtype"]
        )

        print("-" * 70)


    # ========================================================
    # PREDICT ONE SAMPLE
    # ========================================================

    def predict(
        self,
        input_data
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        # Batch size MUST remain 1.
        # ----------------------------------------------------

        input_data = np.asarray(
            input_data,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if input_data.shape != (
            1,
            MAX_FRAMES,
            NUM_MFCC,
            1
        ):

            raise ValueError(
                "Incorrect input shape: "
                f"{input_data.shape}\n"
                "Expected: "
                f"(1, {MAX_FRAMES}, "
                f"{NUM_MFCC}, 1)"
            )


        # ----------------------------------------------------
        # Set input
        # ----------------------------------------------------

        self.interpreter.set_tensor(
            self.input_index,
            input_data
        )


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        self.interpreter.invoke()


        # ----------------------------------------------------
        # Read output
        # ----------------------------------------------------

        output = self.interpreter.get_tensor(
            self.output_index
        )


        return np.asarray(
            output[0],
            dtype=np.float32
        )


# ============================================================
# 6. LOAD LABEL MAP
# ============================================================

def load_labels():

    if not LABEL_MAP_PATH.exists():

        print(
            "\nWARNING:"
        )

        print(
            "label_map.json was not found."
        )

        print(
            "Predictions will be displayed as "
            "Class 0, Class 1, etc."
        )

        return None


    try:

        with open(
            LABEL_MAP_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        labels = {}

        for key, value in data.items():

            labels[int(key)] = str(
                value
            )


        return labels


    except Exception as e:

        print(
            "Could not read label_map.json:"
        )

        print(e)

        return None


# ============================================================
# 7. AUDIO LOADING
# ============================================================

def load_audio(
    audio_path
):

    """
    Match the training / Pi preprocessing.

    Steps:
        soundfile
        stereo -> mono
        float32
        NaN/Inf removal
        resample -> 16 kHz
    """

    audio, original_sr = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=False
    )


    audio = np.asarray(
        audio,
        dtype=np.float32
    )


    original_shape = audio.shape


    # --------------------------------------------------------
    # Stereo -> mono
    # --------------------------------------------------------

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)


    if len(audio) == 0:

        raise ValueError(
            "The audio file is empty."
        )


    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    was_resampled = False


    if original_sr != TARGET_SAMPLE_RATE:

        target_length = int(
            round(
                len(audio)
                * TARGET_SAMPLE_RATE
                / float(original_sr)
            )
        )


        if target_length <= 0:

            raise ValueError(
                "Invalid resampling length."
            )


        audio = resample(
            audio,
            target_length
        ).astype(np.float32)


        was_resampled = True


    return (
        audio,
        int(original_sr),
        original_shape,
        was_resampled
    )


# ============================================================
# 8. MFCC PROCESSING
# ============================================================

def extract_mfcc(
    audio
):

    """
    EXACT MFCC settings from the model training
    and old Pi inference code.
    """

    features = mfcc(
        audio,
        samplerate=TARGET_SAMPLE_RATE,
        numcep=NUM_MFCC,
        winlen=WINLEN,
        winstep=WINSTEP,
        nfft=NFFT
    )


    features = np.asarray(
        features,
        dtype=np.float32
    )


    frames_before = (
        features.shape[0]
    )


    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    if frames_before < MAX_FRAMES:

        padding = (
            MAX_FRAMES
            - frames_before
        )


        features = np.pad(
            features,
            (
                (0, padding),
                (0, 0)
            ),
            mode="constant"
        )


        action = (
            f"PAD {padding} frames"
        )


    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------

    elif frames_before > MAX_FRAMES:

        removed = (
            frames_before
            - MAX_FRAMES
        )


        features = (
            features[:MAX_FRAMES, :]
        )


        action = (
            f"TRUNCATE {removed} frames"
        )


    else:

        action = (
            "NO padding/truncation"
        )


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)


    # --------------------------------------------------------
    # [500,24]
    #       ->
    # [500,24,1]
    #       ->
    # [1,500,24,1]
    # --------------------------------------------------------

    features = features[
        ...,
        np.newaxis
    ]


    features = features[
        np.newaxis,
        ...
    ]


    return (
        features,
        frames_before,
        action
    )


# ============================================================
# 9. CPU / RAM INFORMATION
# ============================================================

def get_system_usage():

    if not PSUTIL_AVAILABLE:

        return None


    try:

        cpu = psutil.cpu_percent(
            interval=0.05
        )

        ram = psutil.virtual_memory()


        return {
            "cpu_percent": float(cpu),
            "ram_percent": float(
                ram.percent
            ),
            "ram_used_mb": float(
                ram.used
                / 1024
                / 1024
            )
        }


    except Exception:

        return None


# ============================================================
# 10. PROCESS ONE AUDIO FILE
# ============================================================

def prepare_audio(
    audio_path
):

    total_start = time.perf_counter()


    # ========================================================
    # AUDIO LOAD
    # ========================================================

    start = time.perf_counter()


    (
        audio,
        original_sr,
        original_shape,
        was_resampled
    ) = load_audio(
        audio_path
    )


    load_time = (
        time.perf_counter()
        - start
    )


    # ========================================================
    # MFCC
    # ========================================================

    start = time.perf_counter()


    (
        input_data,
        frames_before,
        action
    ) = extract_mfcc(
        audio
    )


    mfcc_time = (
        time.perf_counter()
        - start
    )


    preprocess_time = (
        time.perf_counter()
        - total_start
    )


    information = {

        "original_sample_rate":
            original_sr,

        "original_shape":
            original_shape,

        "original_samples":
            len(audio),

        "final_sample_rate":
            TARGET_SAMPLE_RATE,

        "resampled":
            was_resampled,

        "mfcc_frames_before":
            frames_before,

        "mfcc_action":
            action,

        "input_shape":
            input_data.shape,

        "audio_load_time":
            load_time,

        "mfcc_time":
            mfcc_time,

        "preprocess_time":
            preprocess_time
    }


    return (
        input_data,
        information
    )


# ============================================================
# 11. WARM-UP
# ============================================================

def warmup(
    runner,
    input_data
):

    print(
        f"Warm-up: {WARMUP_RUNS} runs..."
    )


    for _ in range(
        WARMUP_RUNS
    ):

        runner.predict(
            input_data
        )


# ============================================================
# 12. BENCHMARK ONE THREAD CONFIGURATION
# ============================================================

def benchmark_runner(
    runner,
    input_data
):

    # --------------------------------------------------------
    # Warm-up first.
    #
    # This avoids measuring initial interpreter setup.
    # --------------------------------------------------------

    warmup(
        runner,
        input_data
    )


    times = []


    cpu_values = []

    ram_values = []


    print(
        f"Benchmarking "
        f"{BENCHMARK_RUNS} "
        f"inference runs..."
    )


    for i in range(
        BENCHMARK_RUNS
    ):

        usage_before = (
            get_system_usage()
        )


        start = time.perf_counter()


        runner.predict(
            input_data
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        times.append(
            elapsed
        )


        if usage_before is not None:

            cpu_values.append(
                usage_before[
                    "cpu_percent"
                ]
            )

            ram_values.append(
                usage_before[
                    "ram_percent"
                ]
            )


        print(
            f"\rRun "
            f"{i + 1:02d}/"
            f"{BENCHMARK_RUNS}"
            f"   "
            f"{elapsed * 1000:.2f} ms",
            end=""
        )


    print()


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    times_ms = [
        x * 1000
        for x in times
    ]


    result = {

        "threads":
            runner.num_threads,

        "runs":
            len(times_ms),

        "minimum_ms":
            float(min(times_ms)),

        "maximum_ms":
            float(max(times_ms)),

        "average_ms":
            float(
                statistics.mean(
                    times_ms
                )
            ),

        "median_ms":
            float(
                statistics.median(
                    times_ms
                )
            ),

        "std_ms":
            float(
                statistics.stdev(
                    times_ms
                )
            )
            if len(times_ms) > 1
            else 0.0
    }


    if cpu_values:

        result[
            "average_pc_cpu_percent"
        ] = float(
            statistics.mean(
                cpu_values
            )
        )


    if ram_values:

        result[
            "average_pc_ram_percent"
        ] = float(
            statistics.mean(
                ram_values
            )
        )


    return result


# ============================================================
# 13. PREDICTION
# ============================================================

def get_prediction(
    runner,
    input_data,
    labels
):

    output = runner.predict(
        input_data
    )


    # --------------------------------------------------------
    # Make sure output is valid
    # --------------------------------------------------------

    output = np.asarray(
        output,
        dtype=np.float32
    )


    if not np.all(
        np.isfinite(output)
    ):

        raise RuntimeError(
            "Model returned NaN/Inf."
        )


    # --------------------------------------------------------
    # Current model is softmax.
    #
    # Normally values already sum to ~1.
    # --------------------------------------------------------

    total = float(
        np.sum(output)
    )


    if total > 0:

        probabilities = (
            output / total
        )

    else:

        probabilities = output


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    if labels is not None:

        predicted_name = labels.get(
            predicted_index,
            f"Class {predicted_index}"
        )

    else:

        predicted_name = (
            f"Class {predicted_index}"
        )


    return (
        predicted_index,
        predicted_name,
        confidence,
        probabilities
    )


# ============================================================
# 14. DISPLAY PREDICTION
# ============================================================

def print_prediction(
    prediction
):

    (
        predicted_index,
        predicted_name,
        confidence,
        probabilities,
        labels
    ) = prediction


    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)


    print(
        f"Class index : {predicted_index}"
    )


    print(
        f"Class       : {predicted_name}"
    )


    print(
        f"Confidence  : "
        f"{confidence * 100:.2f}%"
    )


    print()
    print(
        "All class probabilities:"
    )


    sorted_indices = np.argsort(
        probabilities
    )[::-1]


    for index in sorted_indices:

        if labels is not None:

            name = labels.get(
                int(index),
                f"Class {index}"
            )

        else:

            name = (
                f"Class {index}"
            )


        print(
            f"  {name:<25} "
            f"{probabilities[index] * 100:7.2f}%"
        )


# ============================================================
# 15. PRINT AUDIO INFORMATION
# ============================================================

def print_audio_info(
    info
):

    print()
    print("-" * 70)
    print("AUDIO / MFCC PROCESSING")
    print("-" * 70)


    print(
        "Original sample rate :",
        f"{info['original_sample_rate']} Hz"
    )


    print(
        "Original shape       :",
        info["original_shape"]
    )


    print(
        "Original samples     :",
        info["original_samples"]
    )


    print(
        "Final sample rate    :",
        f"{info['final_sample_rate']} Hz"
    )


    print(
        "Resampled            :",
        "YES"
        if info["resampled"]
        else "NO"
    )


    print(
        "MFCC frames          :",
        info["mfcc_frames_before"]
    )


    print(
        "MFCC action          :",
        info["mfcc_action"]
    )


    print(
        "Final model input    :",
        info["input_shape"]
    )


    print(
        "MFCC parameters      :"
    )


    print(
        f"  numcep = {NUM_MFCC}"
    )


    print(
        f"  winlen = {WINLEN}"
    )


    print(
        f"  winstep = {WINSTEP}"
    )


    print(
        f"  nfft = {NFFT}"
    )


# ============================================================
# 16. BENCHMARK ALL PI THREAD MODES
# ============================================================

def benchmark_all_threads(
    input_data
):

    print()
    print("=" * 70)
    print("PI ZERO 2 W THREAD BENCHMARK")
    print("=" * 70)


    results = []


    for thread_count in THREAD_MODES:

        print()
        print(
            "#" * 70
        )


        print(
            f"# TESTING {thread_count} "
            f"TFLITE THREAD(S)"
        )


        print(
            "#" * 70
        )


        runner = TFLiteRunner(
            MODEL_PATH,
            thread_count
        )


        runner.print_info()


        result = benchmark_runner(
            runner,
            input_data
        )


        results.append(
            result
        )


        print()
        print(
            f"RESULT "
            f"({thread_count} thread):"
        )


        print(
            f"  Minimum : "
            f"{result['minimum_ms']:.2f} ms"
        )


        print(
            f"  Average : "
            f"{result['average_ms']:.2f} ms"
        )


        print(
            f"  Median  : "
            f"{result['median_ms']:.2f} ms"
        )


        print(
            f"  Maximum : "
            f"{result['maximum_ms']:.2f} ms"
        )


        print(
            f"  Std dev : "
            f"{result['std_ms']:.2f} ms"
        )


        # ----------------------------------------------------
        # Clean interpreter before next test.
        # ----------------------------------------------------

        del runner

        gc.collect()


    return results


# ============================================================
# 17. PRINT THREAD COMPARISON
# ============================================================

def print_thread_comparison(
    results
):

    print()
    print("=" * 70)
    print("THREAD PERFORMANCE COMPARISON")
    print("=" * 70)


    print()

    print(
        f"{'Threads':<10}"
        f"{'Min':>12}"
        f"{'Average':>12}"
        f"{'Median':>12}"
        f"{'Max':>12}"
    )


    print(
        "-" * 58
    )


    for result in results:

        print(
            f"{result['threads']:<10}"
            f"{result['minimum_ms']:>11.2f}"
            f"{result['average_ms']:>11.2f}"
            f"{result['median_ms']:>11.2f}"
            f"{result['maximum_ms']:>11.2f}"
        )


    # --------------------------------------------------------
    # Find fastest average.
    # --------------------------------------------------------

    best = min(
        results,
        key=lambda x:
        x["average_ms"]
    )


    print()
    print(
        "Fastest average:"
    )


    print(
        f"  {best['threads']} thread(s)"
        f" -> "
        f"{best['average_ms']:.2f} ms"
    )


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BABY AUDIO - RASPBERRY PI ZERO 2 W TEST")
    print("=" * 70)


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        print()
        print(
            "ERROR:"
        )

        print(
            "TFLite model was not found:"
        )

        print(
            MODEL_PATH.resolve()
        )

        print()
        print(
            "Expected:"
        )

        print(
            r"models\baby_audio_strong.tflite"
        )

        sys.exit(1)


    # ========================================================
    # LOAD LABELS
    # ========================================================

    labels = load_labels()


    # ========================================================
    # SHOW HARDWARE INTENTION
    # ========================================================

    print()
    print(
        "Test configuration:"
    )


    print(
        "  CPU only             : YES"
    )


    print(
        "  GPU                  : DISABLED"
    )


    print(
        "  Batch size           : 1"
    )


    print(
        "  MFCC                 : python_speech_features"
    )


    print(
        "  Sample rate          : 16000 Hz"
    )


    print(
        "  MFCC count           : 24"
    )


    print(
        "  MFCC frames          : 500"
    )


    print(
        "  TFLite thread modes  : 1 / 2 / 4"
    )


    print(
        "  Benchmark runs       :",
        BENCHMARK_RUNS
    )


    # ========================================================
    # INTERACTIVE LOOP
    # ========================================================

    while True:

        print()
        print("=" * 70)
        print(
            "ENTER AUDIO FILE PATH"
        )
        print("=" * 70)


        print(
            r'Example: C:\test\audio.wav'
        )


        print(
            r'Or: "C:\My Audio\test.wav"'
        )


        print(
            "Type EXIT to close."
        )


        try:

            user_input = input(
                "\nAudio path > "
            ).strip()


        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nExiting."
            )

            break


        if not user_input:

            continue


        if user_input.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "Exiting."
            )

            break


        # ----------------------------------------------------
        # Remove surrounding quotes
        # ----------------------------------------------------

        if (
            len(user_input) >= 2
            and user_input[0] == '"'
            and user_input[-1] == '"'
        ):

            user_input = (
                user_input[1:-1]
            )


        audio_path = Path(
            user_input
        )


        # ====================================================
        # CHECK FILE
        # ====================================================

        if not audio_path.exists():

            print()
            print(
                "ERROR: Audio file not found:"
            )

            print(
                audio_path
            )

            continue


        if not audio_path.is_file():

            print()
            print(
                "ERROR: This is not a file:"
            )

            print(
                audio_path
            )

            continue


        # ====================================================
        # PROCESS AUDIO
        # ====================================================

        try:

            print()
            print(
                "=" * 70
            )

            print(
                "PROCESSING:"
            )

            print(
                audio_path
            )


            start_total = (
                time.perf_counter()
            )


            (
                input_data,
                audio_info
            ) = prepare_audio(
                audio_path
            )


            preparation_time = (
                time.perf_counter()
                - start_total
            )


            # -----------------------------------------------
            # Display processing information
            # -----------------------------------------------

            print_audio_info(
                audio_info
            )


            print()
            print(
                f"Total PC preprocessing: "
                f"{preparation_time * 1000:.2f} ms"
            )


            # =================================================
            # FIRST PREDICTION
            # =================================================

            print()
            print(
                "=" * 70
            )

            print(
                "MODEL PREDICTION"
            )

            print(
                "=" * 70
            )


            # Use 4 threads for the displayed prediction.
            #
            # The benchmark below will test all 1/2/4 modes.
            prediction_runner = (
                TFLiteRunner(
                    MODEL_PATH,
                    4
                )
            )


            prediction_runner.print_info()


            # -------------------------------------------------
            # Warmup
            # -------------------------------------------------

            print(
                "\nPerforming warm-up..."
            )


            warmup(
                prediction_runner,
                input_data
            )


            # -------------------------------------------------
            # Actual prediction
            # -------------------------------------------------

            inference_start = (
                time.perf_counter()
            )


            prediction_output = (
                get_prediction(
                    prediction_runner,
                    input_data,
                    labels
                )
            )


            inference_time = (
                time.perf_counter()
                - inference_start
            )


            prediction = (
                prediction_output[0],
                prediction_output[1],
                prediction_output[2],
                prediction_output[3],
                labels
            )


            print_prediction(
                prediction
            )


            print()
            print(
                f"4-thread inference: "
                f"{inference_time * 1000:.2f} ms"
            )


            # =================================================
            # BENCHMARK 1 / 2 / 4 THREADS
            # =================================================

            benchmark_results = (
                benchmark_all_threads(
                    input_data
                )
            )


            print_thread_comparison(
                benchmark_results
            )


            # =================================================
            # END-TO-END ESTIMATION
            # =================================================

            print()
            print("=" * 70)
            print("END-TO-END PROCESSING")
            print("=" * 70)


            best_thread_result = min(
                benchmark_results,
                key=lambda x:
                x["average_ms"]
            )


            best_inference_ms = (
                best_thread_result[
                    "average_ms"
                ]
            )


            end_to_end_ms = (
                preparation_time * 1000
                + best_inference_ms
            )


            print(
                f"Audio loading + MFCC : "
                f"{preparation_time * 1000:.2f} ms"
            )


            print(
                f"Best TFLite inference : "
                f"{best_inference_ms:.2f} ms"
            )


            print(
                f"Estimated total        : "
                f"{end_to_end_ms:.2f} ms"
            )


            print(
                f"Best thread count      : "
                f"{best_thread_result['threads']}"
            )


            # =================================================
            # CLEANUP
            # =================================================

            del prediction_runner

            del input_data

            gc.collect()


            # =================================================
            # FINAL MESSAGE
            # =================================================

            print()
            print("=" * 70)
            print("TEST COMPLETE")
            print("=" * 70)


            print(
                f"Predicted: "
                f"{prediction[1]}"
            )


            print(
                f"Confidence: "
                f"{prediction[2] * 100:.2f}%"
            )


            print(
                f"Best average inference: "
                f"{best_inference_ms:.2f} ms"
            )


            print("=" * 70)


        except Exception as e:

            print()
            print("=" * 70)
            print("ERROR")
            print("=" * 70)

            print(
                str(e)
            )

            print()

            import traceback

            traceback.print_exc()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()