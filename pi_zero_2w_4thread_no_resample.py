# ============================================================
# BABY AUDIO - RASPBERRY PI ZERO 2 W - 4 THREAD TESTER
#
# Uses:
#   - Existing TFLite model
#   - python_speech_features MFCC
#   - Input WAV is ALREADY recorded at 16 kHz
#   - NO resampling
#   - CPU only
#   - EXACTLY 4 TFLite threads
#   - Batch size = 1
#
# Expected input:
#   16 kHz WAV
#   mono or stereo (stereo is converted to mono only)
#
# Model:
#   models/baby_audio_strong.tflite
#
# Label map:
#   models/label_map.json
# ============================================================

import os

# ============================================================
# 1. HARDWARE LIMITS
# ============================================================

# Never use GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit numerical libraries to 4 CPU threads.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "0"


# ============================================================
# 2. IMPORTS
# ============================================================

import sys
import time
import json
import gc
from pathlib import Path

import numpy as np
import soundfile as sf

from python_speech_features import mfcc


# ============================================================
# 3. OPTIONAL PSUTIL
# ============================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================
# 4. SETTINGS
# ============================================================

MODEL_PATH = Path("models") / "baby_audio_strong.tflite"
LABEL_MAP_PATH = Path("models") / "label_map.json"

# IMPORTANT:
# The uploaded audio is already recorded at this rate.
TARGET_SAMPLE_RATE = 16000

NUM_MFCC = 24
WINLEN = 0.025       # 25 ms
WINSTEP = 0.01       # 10 ms
NFFT = 1024
MAX_FRAMES = 500

# Raspberry Pi Zero 2 W = 4 CPU cores.
# This program uses ONLY 4 TFLite threads.
TFLITE_THREADS = 4

WARMUP_RUNS = 3
BENCHMARK_RUNS = 20


# ============================================================
# 5. OPTIONAL PROCESS CPU AFFINITY
# ============================================================

def limit_process_to_four_cpus():
    """
    Restrict this Python process to at most four logical CPUs.

    This is only a PC-side resource limit. It does not emulate
    the ARM Cortex-A53 architecture.
    """
    if not PSUTIL_AVAILABLE:
        print("psutil not installed - CPU affinity not changed.")
        return

    try:
        process = psutil.Process()
        available = process.cpu_affinity()

        if len(available) > 4:
            process.cpu_affinity(available[:4])

        print("CPU affinity:", process.cpu_affinity())

    except Exception as e:
        print("CPU affinity could not be set:", e)


# ============================================================
# 6. LOAD LABELS
# ============================================================

def load_labels():
    if not LABEL_MAP_PATH.exists():
        print("WARNING: label_map.json not found.")
        return None

    try:
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {int(k): str(v) for k, v in data.items()}

    except Exception as e:
        print("Could not read label_map.json:", e)
        return None


# ============================================================
# 7. LOAD TFLITE MODEL
# ============================================================

class TFLiteRunner:

    def __init__(self, model_path):
        self.model_path = str(model_path)

        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.input_index = None
        self.output_index = None

        self.load()

    def load(self):

        print()
        print("Loading TFLite model...")
        print("Threads:", TFLITE_THREADS)

        # Raspberry Pi preferred runtime.
        try:
            from tflite_runtime.interpreter import Interpreter

            print("Runtime: tflite_runtime")

            self.interpreter = Interpreter(
                model_path=self.model_path,
                num_threads=TFLITE_THREADS
            )

        except ImportError:

            # PC fallback.
            try:
                import tensorflow as tf

                print("Runtime: TensorFlow Lite")

                self.interpreter = tf.lite.Interpreter(
                    model_path=self.model_path,
                    num_threads=TFLITE_THREADS
                )

            except ImportError:
                raise RuntimeError(
                    "Neither tflite_runtime nor TensorFlow is installed."
                )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

    def print_info(self):

        inp = self.input_details[0]
        out = self.output_details[0]

        print()
        print("-" * 70)
        print("MODEL INFORMATION")
        print("-" * 70)
        print("TFLite threads :", TFLITE_THREADS)
        print("Input shape    :", inp["shape"])
        print("Input dtype    :", inp["dtype"])
        print("Output shape   :", out["shape"])
        print("Output dtype   :", out["dtype"])

        if "quantization" in inp:
            print("Input quant.   :", inp["quantization"])

        if "quantization" in out:
            print("Output quant.  :", out["quantization"])

        print("-" * 70)

    def predict(self, input_data):

        input_data = np.asarray(
            input_data,
            dtype=np.float32
        )

        expected_shape = (
            1,
            MAX_FRAMES,
            NUM_MFCC,
            1
        )

        if input_data.shape != expected_shape:
            raise ValueError(
                f"Incorrect input shape: {input_data.shape}\n"
                f"Expected: {expected_shape}"
            )

        inp = self.input_details[0]

        # Float model.
        if inp["dtype"] == np.float32:
            model_input = input_data

        # Support int8/uint8 models too.
        else:
            scale, zero_point = inp.get(
                "quantization",
                (0.0, 0)
            )

            if scale == 0:
                raise ValueError(
                    "Quantized model has zero input scale."
                )

            model_input = (
                input_data / scale + zero_point
            )

            if inp["dtype"] == np.int8:
                model_input = np.clip(
                    model_input,
                    -128,
                    127
                ).astype(np.int8)

            elif inp["dtype"] == np.uint8:
                model_input = np.clip(
                    model_input,
                    0,
                    255
                ).astype(np.uint8)

            else:
                model_input = model_input.astype(
                    inp["dtype"]
                )

        self.interpreter.set_tensor(
            self.input_index,
            model_input
        )

        start = time.perf_counter()

        self.interpreter.invoke()

        elapsed = time.perf_counter() - start

        output = self.interpreter.get_tensor(
            self.output_index
        )[0]

        output = np.asarray(
            output,
            dtype=np.float32
        )

        # Dequantize output if necessary.
        out = self.output_details[0]

        if out["dtype"] != np.float32:

            scale, zero_point = out.get(
                "quantization",
                (0.0, 0)
            )

            if scale:
                output = (
                    output.astype(np.float32)
                    - zero_point
                ) * scale

        return output, elapsed


# ============================================================
# 8. LOAD ALREADY-RECORDED AUDIO
# ============================================================

def load_audio(audio_path):
    """
    IMPORTANT:
    No resampling is performed here.

    The input file MUST already be 16 kHz.

    Only:
        WAV -> float32
        stereo -> mono
        NaN/Inf cleanup
    """

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=False
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    original_shape = audio.shape

    # Stereo -> mono.
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # NO CONVERSION / NO RESAMPLING.
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Wrong sample rate: {sample_rate} Hz\n"
            f"This program does NOT resample audio.\n"
            f"Please provide an already-recorded "
            f"{TARGET_SAMPLE_RATE} Hz WAV file."
        )

    return (
        audio,
        int(sample_rate),
        original_shape
    )


# ============================================================
# 9. MFCC FEATURE EXTRACTION
# ============================================================

def extract_mfcc(audio):

    """
    New feature processing:

        python_speech_features.mfcc

    Parameters:
        sample rate = 16000
        MFCC        = 24
        window      = 25 ms
        step        = 10 ms
        FFT         = 1024
        frames      = 500
    """

    # DC removal.
    audio = audio - np.mean(audio)

    # Peak normalization.
    peak = np.max(np.abs(audio))

    if peak > 1e-8:
        audio = audio / peak

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

    frames_before = features.shape[0]

    # Exactly 500 frames.
    if frames_before < MAX_FRAMES:

        padding = MAX_FRAMES - frames_before

        features = np.pad(
            features,
            (
                (0, padding),
                (0, 0)
            ),
            mode="constant"
        )

        action = f"PAD {padding} frames"

    elif frames_before > MAX_FRAMES:

        removed = frames_before - MAX_FRAMES

        features = features[:MAX_FRAMES, :]

        action = f"TRUNCATE {removed} frames"

    else:

        action = "NO padding/truncation"

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    # [500, 24]
    #     ->
    # [500, 24, 1]
    #     ->
    # [1, 500, 24, 1]

    features = features[..., np.newaxis]
    features = features[np.newaxis, ...]

    return (
        features,
        frames_before,
        action
    )


# ============================================================
# 10. PREPARE AUDIO
# ============================================================

def prepare_audio(audio_path):

    start_total = time.perf_counter()

    # --------------------------------------------------------
    # LOAD ONLY
    # --------------------------------------------------------

    start = time.perf_counter()

    (
        audio,
        sample_rate,
        original_shape
    ) = load_audio(audio_path)

    load_time = time.perf_counter() - start

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    start = time.perf_counter()

    (
        input_data,
        frames_before,
        action
    ) = extract_mfcc(audio)

    mfcc_time = time.perf_counter() - start

    total_time = time.perf_counter() - start_total

    info = {
        "sample_rate": sample_rate,
        "original_shape": original_shape,
        "samples": len(audio),
        "duration": len(audio) / sample_rate,
        "frames_before": frames_before,
        "action": action,
        "input_shape": input_data.shape,
        "load_time": load_time,
        "mfcc_time": mfcc_time,
        "total_time": total_time
    }

    return input_data, info


# ============================================================
# 11. PREDICTION
# ============================================================

def get_prediction(runner, input_data, labels):

    output, inference_time = runner.predict(
        input_data
    )

    if not np.all(np.isfinite(output)):
        raise RuntimeError(
            "Model returned NaN/Inf."
        )

    # If output is already softmax, this keeps it unchanged
    # except for tiny numerical differences.
    total = float(np.sum(output))

    if total > 0:
        probabilities = output / total
    else:
        probabilities = output

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
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
        probabilities,
        inference_time
    )


# ============================================================
# 12. PRINT AUDIO INFORMATION
# ============================================================

def print_audio_info(info):

    print()
    print("-" * 70)
    print("AUDIO / NEW MFCC PROCESSING")
    print("-" * 70)

    print(
        "Sample rate          :",
        f"{info['sample_rate']} Hz"
    )

    print(
        "Original shape        :",
        info["original_shape"]
    )

    print(
        "Samples               :",
        info["samples"]
    )

    print(
        "Duration              :",
        f"{info['duration']:.3f} sec"
    )

    print(
        "MFCC frames before    :",
        info["frames_before"]
    )

    print(
        "Frame operation       :",
        info["action"]
    )

    print(
        "Model input           :",
        info["input_shape"]
    )

    print()
    print("MFCC parameters:")
    print("  Sample rate         :", TARGET_SAMPLE_RATE)
    print("  Number of MFCC      :", NUM_MFCC)
    print("  Window              :", WINLEN, "sec")
    print("  Step                :", WINSTEP, "sec")
    print("  FFT                 :", NFFT)
    print("  Max frames          :", MAX_FRAMES)

    print()
    print(
        "Audio load time       :",
        f"{info['load_time'] * 1000:.2f} ms"
    )

    print(
        "MFCC time             :",
        f"{info['mfcc_time'] * 1000:.2f} ms"
    )

    print(
        "Total preprocessing   :",
        f"{info['total_time'] * 1000:.2f} ms"
    )

    print("-" * 70)


# ============================================================
# 13. PRINT PREDICTION
# ============================================================

def print_prediction(
    predicted_index,
    predicted_name,
    confidence,
    probabilities,
    labels,
    inference_time
):

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print(
        "Class index :",
        predicted_index
    )

    print(
        "Class       :",
        predicted_name
    )

    print(
        "Confidence  :",
        f"{confidence * 100:.2f}%"
    )

    print(
        "Inference   :",
        f"{inference_time * 1000:.2f} ms"
    )

    print()
    print("All class probabilities:")

    order = np.argsort(
        probabilities
    )[::-1]

    for index in order:

        if labels is not None:
            name = labels.get(
                int(index),
                f"Class {index}"
            )
        else:
            name = f"Class {index}"

        print(
            f"  {name:<25}"
            f"{probabilities[index] * 100:8.2f}%"
        )

    print("=" * 70)


# ============================================================
# 14. BENCHMARK ONLY 4 THREADS
# ============================================================

def benchmark_4_threads(runner, input_data):

    print()
    print("=" * 70)
    print("4-THREAD TFLITE BENCHMARK")
    print("=" * 70)

    print(
        f"Warm-up runs: {WARMUP_RUNS}"
    )

    for _ in range(WARMUP_RUNS):
        runner.predict(input_data)

    times = []

    print(
        f"Benchmark runs: {BENCHMARK_RUNS}"
    )

    for i in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        _, elapsed = runner.predict(
            input_data
        )

        # Use the interpreter's measured invoke time.
        times.append(elapsed * 1000)

        print(
            f"\rRun {i + 1:02d}/{BENCHMARK_RUNS}"
            f"   {elapsed * 1000:.2f} ms",
            end=""
        )

    print()

    print()
    print("-" * 70)
    print("4-THREAD RESULTS")
    print("-" * 70)

    print(
        "Minimum :",
        f"{min(times):.2f} ms"
    )

    print(
        "Average :",
        f"{np.mean(times):.2f} ms"
    )

    print(
        "Median  :",
        f"{np.median(times):.2f} ms"
    )

    print(
        "Maximum :",
        f"{max(times):.2f} ms"
    )

    if len(times) > 1:
        print(
            "Std dev :",
            f"{np.std(times, ddof=1):.2f} ms"
        )

    print("-" * 70)

    return float(np.mean(times))


# ============================================================
# 15. SYSTEM INFORMATION
# ============================================================

def print_system_info():

    print()
    print("=" * 70)
    print("HARDWARE LIMIT")
    print("=" * 70)

    print("GPU                   : DISABLED")
    print("TFLite CPU threads    :", TFLITE_THREADS)
    print("Batch size            : 1")
    print("MFCC                  : python_speech_features")
    print("Input conversion      : NONE")
    print("Input sample rate     : 16000 Hz REQUIRED")
    print("CPU cores requested   : 4")

    if PSUTIL_AVAILABLE:
        try:
            print(
                "PC logical CPUs       :",
                psutil.cpu_count(logical=True)
            )
        except Exception:
            pass

    print("=" * 70)


# ============================================================
# 16. MAIN
# ============================================================

def main():

    limit_process_to_four_cpus()

    print()
    print("=" * 70)
    print("BABY AUDIO - PI ZERO 2 W 4-THREAD TEST")
    print("=" * 70)

    print_system_info()

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print()
        print("ERROR: Model not found:")
        print(MODEL_PATH.resolve())
        print()
        print(
            r"Expected: models\baby_audio_strong.tflite"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels = load_labels()

    # --------------------------------------------------------
    # LOAD MODEL ONCE
    # --------------------------------------------------------

    runner = TFLiteRunner(
        MODEL_PATH
    )

    runner.print_info()

    # --------------------------------------------------------
    # INTERACTIVE AUDIO LOOP
    # --------------------------------------------------------

    while True:

        print()
        print("=" * 70)
        print("ENTER ALREADY-RECORDED 16 kHz WAV")
        print("=" * 70)

        print(
            r'Example: C:\test\audio.wav'
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
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {
            "exit",
            "quit",
            "q"
        }:
            print("Exiting.")
            break

        # Remove surrounding quotes.
        if (
            len(user_input) >= 2
            and user_input[0] == '"'
            and user_input[-1] == '"'
        ):
            user_input = user_input[1:-1]

        audio_path = Path(user_input)

        if not audio_path.exists():

            print(
                "ERROR: Audio file not found:"
            )
            print(audio_path)
            continue

        if not audio_path.is_file():

            print(
                "ERROR: Path is not a file:"
            )
            print(audio_path)
            continue

        try:

            # =================================================
            # PREPROCESS
            # =================================================

            print()
            print("=" * 70)
            print("PROCESSING AUDIO")
            print("=" * 70)

            print("File:", audio_path)

            start = time.perf_counter()

            input_data, audio_info = prepare_audio(
                audio_path
            )

            preprocessing_time = (
                time.perf_counter() - start
            )

            print_audio_info(audio_info)

            print(
                "Measured total preprocessing:",
                f"{preprocessing_time * 1000:.2f} ms"
            )

            # =================================================
            # PREDICTION
            # =================================================

            print()
            print("=" * 70)
            print("4-THREAD MODEL PREDICTION")
            print("=" * 70)

            (
                predicted_index,
                predicted_name,
                confidence,
                probabilities,
                inference_time
            ) = get_prediction(
                runner,
                input_data,
                labels
            )

            print_prediction(
                predicted_index,
                predicted_name,
                confidence,
                probabilities,
                labels,
                inference_time
            )

            # =================================================
            # BENCHMARK
            # =================================================

            average_inference_ms = benchmark_4_threads(
                runner,
                input_data
            )

            # =================================================
            # END-TO-END
            # =================================================

            total_ms = (
                preprocessing_time * 1000
                + average_inference_ms
            )

            print()
            print("=" * 70)
            print("END-TO-END")
            print("=" * 70)

            print(
                "Audio load + MFCC :",
                f"{preprocessing_time * 1000:.2f} ms"
            )

            print(
                "4-thread inference:",
                f"{average_inference_ms:.2f} ms"
            )

            print(
                "Estimated total   :",
                f"{total_ms:.2f} ms"
            )

            print()
            print("FINAL RESULT")
            print("-" * 70)

            print(
                "Predicted class :",
                predicted_name
            )

            print(
                "Confidence      :",
                f"{confidence * 100:.2f}%"
            )

            print(
                "TFLite threads  :",
                TFLITE_THREADS
            )

            print(
                "Resampling      : NO"
            )

            print("=" * 70)

            del input_data
            gc.collect()

        except Exception as e:

            print()
            print("=" * 70)
            print("ERROR")
            print("=" * 70)
            print(str(e))
            print()

            import traceback
            traceback.print_exc()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
