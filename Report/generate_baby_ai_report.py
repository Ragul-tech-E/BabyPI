import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample
from python_speech_features import mfcc as psf_mfcc

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    import h5py
except ImportError:
    h5py = None

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

H5_PATH = BASE_DIR / "baby_audio_strong_best.h5"
TFLITE_PATH = BASE_DIR / "baby_audio_strong.tflite"

LABEL_MAP_PATH = BASE_DIR / "label_map.json"
NORMALIZATION_PATH = BASE_DIR / "normalization.json"
HISTORY_PATH = BASE_DIR / "training_history.json"
DEPLOYMENT_PATH = BASE_DIR / "deployment_info.json"
TRAIN_LOG_PATH = BASE_DIR / "training_log.csv"

# Your training code used:
# DATASET_ROOT = Path("combined")
#
# This script is inside:
# BabyPI/models/New folder/
#
# Therefore the normal dataset location is:
# BabyPI/combined/test/
PROJECT_ROOT = BASE_DIR.parent.parent
DATASET_ROOT = PROJECT_ROOT / "combined"
TEST_DIR = DATASET_ROOT / "test"

# If your combined folder is somewhere else, change ONLY TEST_DIR.
# Example:
# TEST_DIR = Path(r"C:\Users\Raghul\Desktop\BabyPI\combined\test")

OUTPUT_DIR = BASE_DIR / "AI_REPORT_IMAGES"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact preprocessing from your training code.
TARGET_SAMPLE_RATE = 16000
NUM_MFCC = 24
WINLEN = 0.025
WINSTEP = 0.01
NFFT = 1024
MAX_FRAMES = 500

AUDIO_EXTENSIONS = {
    ".wav",
    ".wave",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
}

# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_class_names():
    data = load_json(LABEL_MAP_PATH)

    if isinstance(data, dict):
        try:
            pairs = sorted(
                [
                    (int(k), str(v))
                    for k, v in data.items()
                ],
                key=lambda x: x[0],
            )

            if pairs:
                return [name for _, name in pairs]

        except Exception:
            pass

    # Fallback only if label_map.json is missing/corrupt.
    return [
        "Asphyxia",
        "Deaf",
        "Hunger",
        "Normal",
        "Pain",
    ]


CLASS_NAMES = load_class_names()


def save_text_report_image(
    title,
    sections,
    filename,
    figsize=(16, 10),
    fontsize=10,
):
    """
    Saves a text-heavy report page as PNG.
    """

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    ax.axis("off")

    y = 0.97

    ax.text(
        0.03,
        y,
        title,
        fontsize=20,
        fontweight="bold",
        va="top",
    )

    y -= 0.065

    for heading, body in sections:

        ax.text(
            0.03,
            y,
            heading,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

        y -= 0.035

        ax.text(
            0.045,
            y,
            body,
            fontsize=fontsize,
            family="monospace",
            va="top",
            linespacing=1.35,
        )

        line_count = body.count("\n") + 1

        y -= (
            0.035
            + 0.032 * line_count
        )

        if y < 0.05:
            break

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ============================================================
# AUDIO / MFCC
# ============================================================

def load_audio_exact(path):
    """
    Matches the training preprocessing:

    soundfile
    stereo -> mono
    NaN/Inf removal
    resample to 16 kHz using scipy.signal.resample
    """

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1,
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if len(audio) == 0:
        raise ValueError(
            f"Empty audio file: {path}"
        )

    if sr != TARGET_SAMPLE_RATE:

        target_length = int(
            round(
                len(audio)
                * TARGET_SAMPLE_RATE
                / float(sr)
            )
        )

        if target_length <= 0:
            raise ValueError(
                f"Invalid target length: {path}"
            )

        audio = resample(
            audio,
            target_length,
        ).astype(np.float32)

    return audio


def extract_mfcc_exact(path):
    """
    EXACT MFCC configuration from training:

        samplerate = 16000
        numcep     = 24
        winlen     = 0.025
        winstep    = 0.01
        nfft       = 1024

    Then force exactly 500 frames.
    """

    audio = load_audio_exact(path)

    features = psf_mfcc(
        audio,
        samplerate=TARGET_SAMPLE_RATE,
        numcep=NUM_MFCC,
        winlen=WINLEN,
        winstep=WINSTEP,
        nfft=NFFT,
    )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.shape[0] < MAX_FRAMES:

        pad_amount = (
            MAX_FRAMES
            - features.shape[0]
        )

        features = np.pad(
            features,
            (
                (0, pad_amount),
                (0, 0),
            ),
            mode="constant",
        )

    else:

        features = features[
            :MAX_FRAMES,
            :
        ]

    if features.shape != (
        MAX_FRAMES,
        NUM_MFCC,
    ):

        raise RuntimeError(
            f"Unexpected MFCC shape "
            f"{features.shape} for {path}"
        )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Model input:
    # (500, 24, 1)
    return features.astype(
        np.float32
    )[..., np.newaxis]


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_test_files():
    files = []
    labels = []

    if not TEST_DIR.exists():
        return (
            files,
            np.asarray(
                labels,
                dtype=np.int64,
            ),
        )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_dir = (
            TEST_DIR
            / class_name
        )

        if not class_dir.exists():
            continue

        for path in sorted(
            class_dir.rglob("*")
        ):

            if (
                path.is_file()
                and path.suffix.lower()
                in AUDIO_EXTENSIONS
            ):

                files.append(path)
                labels.append(class_index)

    return (
        files,
        np.asarray(
            labels,
            dtype=np.int64,
        ),
    )


# ============================================================
# H5 DIAGNOSTIC
# ============================================================

def inspect_h5():
    result = {
        "exists": H5_PATH.exists(),
        "size_bytes": 0,
        "load_status": "Not attempted",
        "error": "",
        "saved_weight_groups": None,
        "model_config_layers": None,
        "keras_model": None,
    }

    if not H5_PATH.exists():

        result["load_status"] = (
            "H5 file not found"
        )

        return result

    result["size_bytes"] = (
        H5_PATH.stat().st_size
    )

    # --------------------------------------------------------
    # Exact custom layer used by the training code.
    # --------------------------------------------------------

    class FixedStandardization(
        tf.keras.layers.Layer
    ):

        def __init__(
            self,
            mean,
            std,
            **kwargs,
        ):

            super().__init__(
                **kwargs
            )

            self.mean_values = (
                np.asarray(
                    mean,
                    dtype=np.float32,
                ).reshape(
                    1,
                    1,
                    NUM_MFCC,
                    1,
                )
            )

            self.std_values = np.maximum(
                np.asarray(
                    std,
                    dtype=np.float32,
                ).reshape(
                    1,
                    1,
                    NUM_MFCC,
                    1,
                ),
                1e-6,
            )

        def build(
            self,
            input_shape,
        ):

            self.mean_weight = (
                self.add_weight(
                    name="mean",
                    shape=self.mean_values.shape,
                    initializer=tf.keras.initializers.Constant(
                        self.mean_values
                    ),
                    trainable=False,
                )
            )

            self.std_weight = (
                self.add_weight(
                    name="std",
                    shape=self.std_values.shape,
                    initializer=tf.keras.initializers.Constant(
                        self.std_values
                    ),
                    trainable=False,
                )
            )

            super().build(
                input_shape
            )

        def call(
            self,
            inputs,
        ):

            inputs = tf.cast(
                inputs,
                tf.float32,
            )

            return (
                inputs
                - self.mean_weight
            ) / self.std_weight

        def get_config(self):

            config = super().get_config()

            config.update(
                {
                    "mean":
                        self.mean_values.reshape(
                            -1
                        ).tolist(),

                    "std":
                        self.std_values.reshape(
                            -1
                        ).tolist(),
                }
            )

            return config

    # --------------------------------------------------------
    # Try to load H5.
    # --------------------------------------------------------

    try:

        model = (
            tf.keras.models.load_model(
                str(H5_PATH),
                custom_objects={
                    "FixedStandardization":
                        FixedStandardization,

                    "Custom>FixedStandardization":
                        FixedStandardization,
                },
                compile=False,
            )
        )

        result["load_status"] = (
            "Loaded successfully"
        )

        result["keras_model"] = model

        return result

    except Exception as e:

        result["load_status"] = (
            "FAILED"
        )

        result["error"] = str(e)

    # --------------------------------------------------------
    # Inspect H5 without loading the Keras model.
    # --------------------------------------------------------

    if h5py is not None:

        try:

            with h5py.File(
                H5_PATH,
                "r",
            ) as f:

                if (
                    "model_weights"
                    in f
                ):

                    result[
                        "saved_weight_groups"
                    ] = len(
                        f[
                            "model_weights"
                        ].keys()
                    )

                config = f.attrs.get(
                    "model_config"
                )

                if config is not None:

                    if isinstance(
                        config,
                        bytes,
                    ):
                        config = (
                            config.decode(
                                "utf-8"
                            )
                        )

                    cfg = json.loads(
                        config
                    )

                    layers_config = (
                        cfg
                        .get(
                            "config",
                            {},
                        )
                        .get(
                            "layers",
                            [],
                        )
                    )

                    result[
                        "model_config_layers"
                    ] = len(
                        layers_config
                    )

        except Exception as e:

            result[
                "h5_inspection_error"
            ] = str(e)

    return result


# ============================================================
# TFLITE
# ============================================================

def load_tflite():

    if not TFLITE_PATH.exists():

        raise FileNotFoundError(
            f"TFLite model not found:\n"
            f"{TFLITE_PATH}"
        )

    interpreter = (
        tf.lite.Interpreter(
            model_path=str(
                TFLITE_PATH
            ),
            num_threads=4,
        )
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter
        .get_input_details()
    )

    output_details = (
        interpreter
        .get_output_details()
    )

    return (
        interpreter,
        input_details[0],
        output_details[0],
    )


def tflite_predict(
    interpreter,
    input_detail,
    output_detail,
    x,
):
    """
    x shape = (500, 24, 1)
    """

    sample = x[
        np.newaxis,
        ...
    ].astype(
        np.float32
    )

    # Handle fixed/dynamic batch shape.
    current_shape = (
        input_detail["shape"]
    )

    if list(current_shape) != list(
        sample.shape
    ):

        try:

            interpreter.resize_tensor_input(
                input_detail["index"],
                sample.shape,
                strict=False,
            )

            interpreter.allocate_tensors()

            input_detail = (
                interpreter
                .get_input_details()[0]
            )

            output_detail = (
                interpreter
                .get_output_details()[0]
            )

        except Exception:
            pass

    # Normally this model is FLOAT32.
    # Quantized input is also handled safely.
    if (
        input_detail["dtype"]
        != np.float32
    ):

        if np.issubdtype(
            input_detail["dtype"],
            np.integer,
        ):

            scale, zero_point = (
                input_detail.get(
                    "quantization",
                    (0.0, 0),
                )
            )

            if scale:

                sample = np.round(
                    sample / scale
                    + zero_point
                ).astype(
                    input_detail[
                        "dtype"
                    ]
                )

            else:

                sample = sample.astype(
                    input_detail[
                        "dtype"
                    ]
                )

        else:

            sample = sample.astype(
                input_detail["dtype"]
            )

    interpreter.set_tensor(
        input_detail["index"],
        sample,
    )

    interpreter.invoke()

    output = (
        interpreter.get_tensor(
            output_detail["index"]
        )[0]
    )

    # Dequantize output if required.
    if (
        output_detail["dtype"]
        != np.float32
        and np.issubdtype(
            output_detail["dtype"],
            np.integer,
        )
    ):

        scale, zero_point = (
            output_detail.get(
                "quantization",
                (0.0, 0),
            )
        )

        if scale:

            output = (
                output.astype(
                    np.float32
                )
                - zero_point
            ) * scale

    return (
        np.asarray(
            output,
            dtype=np.float32,
        ),
        input_detail,
        output_detail,
    )


def evaluate_tflite(
    test_files,
    y_true,
):

    (
        interpreter,
        input_detail,
        output_detail,
    ) = load_tflite()

    probabilities = []
    predictions = []

    good_files = []
    failures = []

    start_time = time.time()

    total = len(test_files)

    for i, path in enumerate(
        test_files
    ):

        try:

            features = (
                extract_mfcc_exact(
                    path
                )
            )

            output, input_detail, output_detail = (
                tflite_predict(
                    interpreter,
                    input_detail,
                    output_detail,
                    features,
                )
            )

            probabilities.append(
                output
            )

            predictions.append(
                int(
                    np.argmax(
                        output
                    )
                )
            )

            good_files.append(
                path
            )

        except Exception as e:

            failures.append(
                {
                    "file":
                        str(path),

                    "error":
                        str(e),
                }
            )

        if (
            (i + 1) % 10 == 0
            or i == total - 1
        ):

            print(
                f"\rEvaluating test audio: "
                f"{i + 1}/{total}",
                end="",
            )

    print()

    if not predictions:

        raise RuntimeError(
            "No test files could be evaluated."
        )

    # Keep labels aligned with successfully evaluated files.
    good_indices = [
        i
        for i, p in enumerate(
            test_files
        )
        if p in good_files
    ]

    y_eval = y_true[
        good_indices
    ]

    return {
        "probabilities":
            np.asarray(
                probabilities,
                dtype=np.float32,
            ),

        "predictions":
            np.asarray(
                predictions,
                dtype=np.int64,
            ),

        "y_true":
            np.asarray(
                y_eval,
                dtype=np.int64,
            ),

        "files":
            good_files,

        "failures":
            failures,

        "seconds":
            time.time()
            - start_time,

        "input_detail":
            input_detail,

        "output_detail":
            output_detail,
    }


# ============================================================
# TFLITE TENSOR REPORT
# ============================================================

def get_tflite_tensor_report():

    (
        interpreter,
        input_detail,
        output_detail,
    ) = load_tflite()

    details = (
        interpreter
        .get_tensor_details()
    )

    rows = []

    for index, detail in enumerate(
        details
    ):

        shape = tuple(
            int(v)
            for v in detail.get(
                "shape",
                [],
            )
        )

        rows.append(
            (
                index,
                detail.get(
                    "name",
                    "",
                ),
                shape,
                str(
                    detail.get(
                        "dtype",
                        "",
                    )
                ),
            )
        )

    return (
        rows,
        input_detail,
        output_detail,
    )


# ============================================================
# IMAGE REPORTS
# ============================================================

def make_bar_chart(
    labels,
    values,
    title,
    ylabel,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(fig)

    return path


def make_confusion_matrix(
    cm,
    names,
    title,
    filename,
    normalize=False,
):

    data = cm.astype(
        np.float64
    )

    if normalize:

        row_sum = (
            data.sum(
                axis=1,
                keepdims=True,
            )
        )

        data = np.divide(
            data,
            row_sum,
            out=np.zeros_like(
                data
            ),
            where=row_sum != 0,
        )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        data
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ax.set_xticks(
        range(len(names))
    )

    ax.set_xticklabels(
        names,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        range(len(names))
    )

    ax.set_yticklabels(
        names
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
    )

    threshold = (
        data.max() / 2
        if data.size
        else 0
    )

    for i in range(
        data.shape[0]
    ):

        for j in range(
            data.shape[1]
        ):

            if normalize:

                text = (
                    f"{data[i, j]:.2f}"
                )

            else:

                text = (
                    f"{int(data[i, j])}"
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if data[i, j]
                    > threshold
                    else "black"
                ),
                fontsize=11,
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(fig)

    return path


def make_training_history(
    history
):

    if not history:
        return []

    output_paths = []

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if (
        "accuracy" in history
        or "val_accuracy" in history
    ):

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        if "accuracy" in history:

            ax.plot(
                history["accuracy"],
                label="Training accuracy",
            )

        if "val_accuracy" in history:

            ax.plot(
                history["val_accuracy"],
                label="Validation accuracy",
            )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Accuracy"
        )

        ax.set_title(
            "Training / Validation Accuracy",
            fontsize=16,
            fontweight="bold",
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        path = (
            OUTPUT_DIR
            / "07_training_accuracy.png"
        )

        fig.savefig(
            path,
            dpi=200,
        )

        plt.close(fig)

        output_paths.append(
            path
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    if (
        "loss" in history
        or "val_loss" in history
    ):

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        if "loss" in history:

            ax.plot(
                history["loss"],
                label="Training loss",
            )

        if "val_loss" in history:

            ax.plot(
                history["val_loss"],
                label="Validation loss",
            )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Loss"
        )

        ax.set_title(
            "Training / Validation Loss",
            fontsize=16,
            fontweight="bold",
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        path = (
            OUTPUT_DIR
            / "08_training_loss.png"
        )

        fig.savefig(
            path,
            dpi=200,
        )

        plt.close(fig)

        output_paths.append(
            path
        )

    return output_paths


def make_mfcc_example_images(
    test_files,
    max_examples=6,
):

    """
    Creates separate MFCC images for the first few
    test samples. Each sample gets its own PNG.
    """

    paths = []

    for i, path in enumerate(
        test_files[:max_examples]
    ):

        try:

            features = (
                extract_mfcc_exact(
                    path
                )
            )

            # Remove channel dimension.
            matrix = features[
                :, :, 0
            ]

            fig, ax = plt.subplots(
                figsize=(12, 6)
            )

            image = ax.imshow(
                matrix.T,
                aspect="auto",
                origin="lower",
            )

            fig.colorbar(
                image,
                ax=ax,
                label="MFCC value",
            )

            ax.set_xlabel(
                "Frame"
            )

            ax.set_ylabel(
                "MFCC coefficient"
            )

            ax.set_title(
                "Test Audio MFCC\n"
                + path.name,
                fontsize=15,
                fontweight="bold",
            )

            fig.tight_layout()

            output_path = (
                OUTPUT_DIR
                / (
                    f"16_test_mfcc_"
                    f"{i+1:02d}.png"
                )
            )

            fig.savefig(
                output_path,
                dpi=200,
            )

            plt.close(fig)

            paths.append(
                output_path
            )

        except Exception:
            pass

    return paths


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "BABY CRY AI — COMPLETE REPORT GENERATOR"
    )
    print("=" * 80)

    print(
        "\nBase folder:"
    )

    print(
        BASE_DIR
    )

    print(
        "\nExpected test folder:"
    )

    print(
        TEST_DIR
    )

    # --------------------------------------------------------
    # H5 inspection
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "CHECKING H5 MODEL"
    )

    print(
        "=" * 80
    )

    h5_info = inspect_h5()

    print(
        "\nH5 status:"
    )

    print(
        h5_info["load_status"]
    )

    if h5_info["error"]:

        print(
            "\nH5 loading error:"
        )

        print(
            h5_info["error"]
        )

    # --------------------------------------------------------
    # TFLite
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "LOADING TFLITE DEPLOYMENT MODEL"
    )

    print(
        "=" * 80
    )

    (
        tensor_rows,
        tflite_input,
        tflite_output,
    ) = get_tflite_tensor_report()

    tflite_size_mb = (
        TFLITE_PATH.stat().st_size
        / 1024
        / 1024
    )

    print(
        "\nTFLite loaded successfully."
    )

    print(
        "Input shape:",
        tflite_input["shape"],
    )

    print(
        "Input dtype:",
        tflite_input["dtype"],
    )

    print(
        "Output shape:",
        tflite_output["shape"],
    )

    print(
        "Output dtype:",
        tflite_output["dtype"],
    )

    print(
        "Tensor count:",
        len(tensor_rows),
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FINDING TEST DATA"
    )

    print(
        "=" * 80
    )

    test_files, y_true = (
        find_test_files()
    )

    print(
        "\nTest audio files:",
        len(test_files),
    )

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(
                y_true == i
            )
        )

        print(
            f"  {i}: "
            f"{class_name:<25} "
            f"{count}"
        )

    # --------------------------------------------------------
    # TFLite test evaluation
    # --------------------------------------------------------

    evaluation = None

    if len(test_files) > 0:

        print(
            "\n" + "=" * 80
        )

        print(
            "RUNNING FINAL TEST EVALUATION"
        )

        print(
            "=" * 80
        )

        evaluation = (
            evaluate_tflite(
                test_files,
                y_true,
            )
        )

        probabilities = (
            evaluation[
                "probabilities"
            ]
        )

        predictions = (
            evaluation[
                "predictions"
            ]
        )

        y_eval = (
            evaluation[
                "y_true"
            ]
        )

        evaluated_files = (
            evaluation[
                "files"
            ]
        )

        accuracy = (
            accuracy_score(
                y_eval,
                predictions,
            )
        )

        precision = (
            precision_score(
                y_eval,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )

        recall = (
            recall_score(
                y_eval,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )

        f1 = (
            f1_score(
                y_eval,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )

        cm = confusion_matrix(
            y_eval,
            predictions,
            labels=list(
                range(
                    len(
                        CLASS_NAMES
                    )
                )
            ),
        )

        report_text = (
            classification_report(
                y_eval,
                predictions,
                labels=list(
                    range(
                        len(
                            CLASS_NAMES
                        )
                    )
                ),
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )

        report_dict = (
            classification_report(
                y_eval,
                predictions,
                labels=list(
                    range(
                        len(
                            CLASS_NAMES
                        )
                    )
                ),
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
        )

        print(
            "\nTEST RESULTS"
        )

        print(
            f"Accuracy          : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Weighted precision: "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Weighted recall   : "
            f"{recall * 100:.2f}%"
        )

        print(
            f"Weighted F1       : "
            f"{f1 * 100:.2f}%"
        )

        print(
            "\nClassification report:"
        )

        print(
            report_text
        )

    else:

        probabilities = None
        predictions = None
        y_eval = None
        evaluated_files = []
        accuracy = None
        precision = None
        recall = None
        f1 = None
        cm = None
        report_text = None
        report_dict = None

        print(
            "\nWARNING:"
        )

        print(
            "No test audio was found."
        )

        print(
            "Model reports will still be generated."
        )

    # --------------------------------------------------------
    # Load saved training/deployment reports
    # --------------------------------------------------------

    history = load_json(
        HISTORY_PATH
    )

    normalization = load_json(
        NORMALIZATION_PATH
    )

    deployment = load_json(
        DEPLOYMENT_PATH
    )

    # --------------------------------------------------------
    # Page 1 — Executive summary
    # --------------------------------------------------------

    h5_error_short = (
        h5_info["error"][:600]
        if h5_info["error"]
        else "None"
    )

    accuracy_text = (
        f"{accuracy * 100:.2f}%"
        if accuracy is not None
        else "Not evaluated"
    )

    executive_sections = [

        (
            "MODEL FILES",

            f"H5 best model : "
            f"{H5_PATH.name}\n"

            f"H5 size       : "
            f"{h5_info['size_bytes'] / 1024 / 1024:.3f} MB\n"

            f"TFLite model  : "
            f"{TFLITE_PATH.name}\n"

            f"TFLite size   : "
            f"{tflite_size_mb:.3f} MB"
        ),

        (
            "H5 LOADING",

            f"Status        : "
            f"{h5_info['load_status']}\n"

            f"Saved weight groups : "
            f"{h5_info.get('saved_weight_groups', 'N/A')}\n"

            f"Model-config layers : "
            f"{h5_info.get('model_config_layers', 'N/A')}\n"

            f"Error         : "
            f"{h5_error_short}"
        ),

        (
            "DEPLOYMENT INPUT",

            f"Sample rate   : "
            f"{TARGET_SAMPLE_RATE} Hz\n"

            f"MFCC library  : "
            f"python_speech_features\n"

            f"MFCC count    : "
            f"{NUM_MFCC}\n"

            f"Window        : "
            f"{WINLEN} s\n"

            f"Step          : "
            f"{WINSTEP} s\n"

            f"NFFT          : "
            f"{NFFT}\n"

            f"Frames        : "
            f"{MAX_FRAMES}\n"

            f"Input shape   : "
            f"{tuple(int(v) for v in tflite_input['shape'])}\n"

            f"Input dtype   : "
            f"{tflite_input['dtype']}"
        ),

        (
            "FINAL TEST",

            f"Test folder   : "
            f"{TEST_DIR}\n"

            f"Files found   : "
            f"{len(test_files)}\n"

            f"Files tested  : "
            f"{len(evaluated_files)}\n"

            f"Failures      : "
            f"{len(evaluation['failures']) if evaluation else 0}\n"

            f"Accuracy      : "
            f"{accuracy_text}"
        ),
    ]

    save_text_report_image(
        "BABY CRY AI — EXECUTIVE REPORT",
        executive_sections,
        "01_executive_summary.png",
        figsize=(16, 12),
        fontsize=10,
    )

    # --------------------------------------------------------
    # Page 2 — TFLite architecture/tensors
    # --------------------------------------------------------

    architecture_lines = []

    architecture_lines.append(
        f"TFLite tensor count: "
        f"{len(tensor_rows)}"
    )

    architecture_lines.append(
        f"Output shape: "
        f"{tuple(int(v) for v in tflite_output['shape'])}"
    )

    architecture_lines.append("")

    architecture_lines.append(
        f"{'No.':<5}"
        f"{'Tensor name':<62}"
        f"{'Shape':<30}"
        f"dtype"
    )

    architecture_lines.append(
        "-" * 120
    )

    for (
        index,
        name,
        shape,
        dtype,
    ) in tensor_rows:

        architecture_lines.append(
            f"{index:<5}"
            f"{name[:60]:<62}"
            f"{str(shape):<30}"
            f"{dtype}"
        )

    if (
        h5_info["keras_model"]
        is not None
    ):

        keras_model = (
            h5_info[
                "keras_model"
            ]
        )

        architecture_lines.append(
            ""
        )

        architecture_lines.append(
            f"Keras total parameters: "
            f"{keras_model.count_params():,}"
        )

        architecture_lines.append(
            ""
        )

        architecture_lines.append(
            "Keras layers:"
        )

        for i, layer in enumerate(
            keras_model.layers
        ):

            architecture_lines.append(
                f"{i:<4} "
                f"{layer.name:<35} "
                f"{layer.__class__.__name__:<30} "
                f"{layer.count_params():>12,}"
            )

    else:

        architecture_lines.append(
            ""
        )

        architecture_lines.append(
            "Keras H5 was not used for inference "
            "because it could not be loaded."
        )

        architecture_lines.append(
            "The table above is taken directly "
            "from the working TFLite deployment model."
        )

    save_text_report_image(
        "MODEL ARCHITECTURE / TFLITE TENSORS",
        [
            (
                "MODEL STRUCTURE",
                "\n".join(
                    architecture_lines
                ),
            )
        ],
        "02_model_architecture.png",
        figsize=(18, 15),
        fontsize=7,
    )

    # --------------------------------------------------------
    # Page 3 — Test dataset report
    # --------------------------------------------------------

    dataset_lines = [
        f"Test directory: {TEST_DIR}",
        f"Total test audio files: {len(test_files)}",
        "",
        "CLASS DISTRIBUTION",
        "-" * 55,
    ]

    class_counts = []

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(
                y_true == i
            )
        )

        class_counts.append(
            count
        )

        dataset_lines.append(
            f"{i:>2}  "
            f"{class_name:<25} "
            f"{count:>7}"
        )

    dataset_lines.extend(
        [
            "",
            "PREPROCESSING USED FOR TESTING",
            "-" * 55,
            "Audio -> mono",
            "Resample -> 16,000 Hz",
            "MFCC -> python_speech_features",
            "numcep = 24",
            "winlen = 0.025 s",
            "winstep = 0.01 s",
            "nfft = 1024",
            "Pad/crop -> 500 frames",
            "Final input -> (500, 24, 1)",
            "No second MFCC normalization is applied here.",
            "The trained model contains its fixed standardization layer.",
        ]
    )

    save_text_report_image(
        "TEST DATASET REPORT",
        [
            (
                "TEST DATA",
                "\n".join(
                    dataset_lines
                ),
            )
        ],
        "03_test_dataset.png",
        figsize=(16, 12),
        fontsize=10,
    )

    if test_files:

        make_bar_chart(
            CLASS_NAMES,
            class_counts,
            "Test Dataset Class Distribution",
            "Number of audio files",
            "04_test_class_distribution.png",
        )

    # --------------------------------------------------------
    # Page 5 — Overall test metrics
    # --------------------------------------------------------

    if evaluation is not None:

        evaluation_seconds = (
            evaluation["seconds"]
        )

        metric_lines = (

            f"Accuracy             : "
            f"{accuracy * 100:.2f}%\n"

            f"Weighted precision   : "
            f"{precision * 100:.2f}%\n"

            f"Weighted recall      : "
            f"{recall * 100:.2f}%\n"

            f"Weighted F1          : "
            f"{f1 * 100:.2f}%\n"

            f"\n"

            f"Files evaluated      : "
            f"{len(evaluated_files)}\n"

            f"Evaluation time      : "
            f"{evaluation_seconds:.2f} seconds\n"

            f"Average/file         : "
            f"{evaluation_seconds / max(len(evaluated_files), 1):.4f} seconds"
        )

        save_text_report_image(
            "FINAL TFLITE TEST PERFORMANCE",
            [
                (
                    "OVERALL TEST METRICS",
                    metric_lines,
                ),

                (
                    "INTERPRETATION",
                    "Accuracy, precision, recall and F1 are calculated "
                    "from the test audio found in combined/test. "
                    "The TFLite model is evaluated using the same MFCC "
                    "preprocessing configuration used during training.",
                ),
            ],
            "05_test_metrics.png",
            figsize=(16, 10),
            fontsize=11,
        )

        # ----------------------------------------------------
        # Per-class metrics
        # ----------------------------------------------------

        per_class_lines = [
            f"{'Class':<25}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1':>12}"
            f"{'Support':>12}",
            "-" * 75,
        ]

        f1_values = []

        for class_name in CLASS_NAMES:

            d = report_dict[
                class_name
            ]

            p = (
                d["precision"]
                * 100
            )

            r = (
                d["recall"]
                * 100
            )

            f = (
                d["f1-score"]
                * 100
            )

            s = int(
                d["support"]
            )

            f1_values.append(
                f
            )

            per_class_lines.append(
                f"{class_name:<25}"
                f"{p:>11.2f}%"
                f"{r:>11.2f}%"
                f"{f:>11.2f}%"
                f"{s:>12}"
            )

        save_text_report_image(
            "PER-CLASS TEST PERFORMANCE",
            [
                (
                    "CLASS METRICS",
                    "\n".join(
                        per_class_lines
                    ),
                )
            ],
            "06_per_class_metrics.png",
            figsize=(16, 10),
            fontsize=11,
        )

        make_bar_chart(
            CLASS_NAMES,
            f1_values,
            "Per-Class F1 Score",
            "F1 score (%)",
            "06b_per_class_f1.png",
        )

        # ----------------------------------------------------
        # Confusion matrices
        # ----------------------------------------------------

        make_confusion_matrix(
            cm,
            CLASS_NAMES,
            "TFLite Test Confusion Matrix",
            "09_confusion_matrix.png",
            normalize=False,
        )

        make_confusion_matrix(
            cm,
            CLASS_NAMES,
            "TFLite Test Confusion Matrix — Row Normalized",
            "10_confusion_matrix_normalized.png",
            normalize=True,
        )

        # ----------------------------------------------------
        # Classification report image
        # ----------------------------------------------------

        save_text_report_image(
            "CLASSIFICATION REPORT",
            [
                (
                    "SCIKIT-LEARN CLASSIFICATION REPORT",
                    report_text,
                )
            ],
            "11_classification_report.png",
            figsize=(15, 10),
            fontsize=11,
        )

        # ----------------------------------------------------
        # Sample predictions
        # ----------------------------------------------------

        sample_lines = [
            f"{'Index':<7}"
            f"{'Actual':<25}"
            f"{'Predicted':<25}"
            f"{'Confidence':>12}"
        ]

        sample_lines.append(
            "-" * 75
        )

        for i in range(
            min(
                50,
                len(
                    evaluated_files
                ),
            )
        ):

            actual = CLASS_NAMES[
                int(
                    y_eval[i]
                )
            ]

            predicted_index = int(
                predictions[i]
            )

            predicted = (
                CLASS_NAMES[
                    predicted_index
                ]
            )

            confidence = float(
                np.max(
                    probabilities[i]
                )
            )

            sample_lines.append(
                f"{i:<7}"
                f"{actual:<25}"
                f"{predicted:<25}"
                f"{confidence:>11.2%}"
            )

        save_text_report_image(
            "SAMPLE TEST PREDICTIONS",
            [
                (
                    "FIRST 50 EVALUATED TEST FILES",
                    "\n".join(
                        sample_lines
                    ),
                )
            ],
            "12_sample_predictions.png",
            figsize=(16, 15),
            fontsize=9,
        )

        # ----------------------------------------------------
        # Full test file prediction CSV
        # ----------------------------------------------------

        csv_path = (
            OUTPUT_DIR
            / "test_predictions.csv"
        )

        with open(
            csv_path,
            "w",
            encoding="utf-8",
        ) as f:

            header = (
                "index,file,actual,"
                "predicted,confidence"
            )

            f.write(
                header + "\n"
            )

            for i, path in enumerate(
                evaluated_files
            ):

                actual_index = int(
                    y_eval[i]
                )

                predicted_index = int(
                    predictions[i]
                )

                confidence = float(
                    np.max(
                        probabilities[i]
                    )
                )

                row = (
                    f"{i},"
                    f"\"{str(path).replace(chr(34), chr(34)*2)}\","
                    f"\"{CLASS_NAMES[actual_index]}\","
                    f"\"{CLASS_NAMES[predicted_index]}\","
                    f"{confidence:.6f}\n"
                )

                f.write(
                    row
                )

    # --------------------------------------------------------
    # Training curves
    # --------------------------------------------------------

    history_paths = (
        make_training_history(
            history
        )
    )

    # --------------------------------------------------------
    # Test MFCC images
    # --------------------------------------------------------

    mfcc_paths = (
        make_mfcc_example_images(
            test_files,
            max_examples=6,
        )
    )

    # --------------------------------------------------------
    # Deployment / normalization report
    # --------------------------------------------------------

    deployment_lines = []

    if deployment:

        for key, value in (
            deployment.items()
        ):

            deployment_lines.append(
                f"{key}: {value}"
            )

    else:

        deployment_lines.append(
            "deployment_info.json not found."
        )

    deployment_lines.extend(
        [
            "",
            "NORMALIZATION FILE",
            "-" * 55,
        ]
    )

    if normalization:

        means = normalization.get(
            "mean",
            [],
        )

        stds = normalization.get(
            "std",
            [],
        )

        deployment_lines.append(
            f"Mean values: "
            f"{len(means)}"
        )

        deployment_lines.append(
            f"Std values : "
            f"{len(stds)}"
        )

        if means:

            deployment_lines.append(
                f"First mean : "
                f"{means[0]:.6f}"
            )

        if stds:

            deployment_lines.append(
                f"First std  : "
                f"{stds[0]:.6f}"
            )

        deployment_lines.append(
            "Training configuration states that "
            "normalization is embedded inside the model."
        )

    else:

        deployment_lines.append(
            "normalization.json not found."
        )

    save_text_report_image(
        "DEPLOYMENT / NORMALIZATION REPORT",
        [
            (
                "DEPLOYMENT INFORMATION",
                "\n".join(
                    deployment_lines
                ),
            )
        ],
        "14_deployment_info.png",
        figsize=(17, 14),
        fontsize=9,
    )

    # --------------------------------------------------------
    # H5 diagnostic report
    # --------------------------------------------------------

    diagnosis_lines = [

        "The H5 model was tested with the custom",
        "FixedStandardization layer used by the training code.",

        "",
        f"H5 status: "
        f"{h5_info['load_status']}",

        "",
        "H5 error:",
        h5_info["error"]
        if h5_info["error"]
        else "None",

        "",
        f"H5 saved weight groups: "
        f"{h5_info.get('saved_weight_groups', 'N/A')}",

        f"H5 model-config layers: "
        f"{h5_info.get('model_config_layers', 'N/A')}",

        "",
        "IMPORTANT:",
        "The TFLite model is the working deployment artifact.",
        "This report therefore uses TFLite for the final test",
        "evaluation instead of inventing a replacement Keras model.",
    ]

    save_text_report_image(
        "H5 LOADING DIAGNOSTIC",
        [
            (
                "H5 DIAGNOSTIC",
                "\n".join(
                    diagnosis_lines
                ),
            )
        ],
        "15_h5_diagnostic.png",
        figsize=(17, 12),
        fontsize=10,
    )

    # --------------------------------------------------------
    # Master JSON
    # --------------------------------------------------------

    master_report = {

        "generator":
            "Complete Baby Cry AI Report Generator",

        "base_dir":
            str(BASE_DIR),

        "tensorflow_version":
            str(tf.__version__),

        "classes":
            CLASS_NAMES,

        "h5": {

            "path":
                str(H5_PATH),

            "size_MB":
                h5_info["size_bytes"]
                / 1024
                / 1024,

            "load_status":
                h5_info[
                    "load_status"
                ],

            "error":
                h5_info[
                    "error"
                ],

            "saved_weight_groups":
                h5_info.get(
                    "saved_weight_groups"
                ),

            "model_config_layers":
                h5_info.get(
                    "model_config_layers"
                ),
        },

        "tflite": {

            "path":
                str(TFLITE_PATH),

            "size_MB":
                tflite_size_mb,

            "input_shape":
                [
                    int(v)
                    for v in
                    tflite_input[
                        "shape"
                    ]
                ],

            "input_dtype":
                str(
                    tflite_input[
                        "dtype"
                    ]
                ),

            "output_shape":
                [
                    int(v)
                    for v in
                    tflite_output[
                        "shape"
                    ]
                ],

            "output_dtype":
                str(
                    tflite_output[
                        "dtype"
                    ]
                ),

            "tensor_count":
                len(tensor_rows),
        },

        "preprocessing": {

            "sample_rate":
                TARGET_SAMPLE_RATE,

            "mfcc_library":
                "python_speech_features",

            "num_mfcc":
                NUM_MFCC,

            "winlen":
                WINLEN,

            "winstep":
                WINSTEP,

            "nfft":
                NFFT,

            "max_frames":
                MAX_FRAMES,

            "final_shape":
                [
                    MAX_FRAMES,
                    NUM_MFCC,
                    1,
                ],
        },

        "test": {

            "directory":
                str(TEST_DIR),

            "files_found":
                len(test_files),

            "files_evaluated":
                len(evaluated_files),

            "failures":
                (
                    evaluation[
                        "failures"
                    ]
                    if evaluation
                    else []
                ),

            "accuracy":
                (
                    float(
                        accuracy
                    )
                    if accuracy
                    is not None
                    else None
                ),

            "weighted_precision":
                (
                    float(
                        precision
                    )
                    if precision
                    is not None
                    else None
                ),

            "weighted_recall":
                (
                    float(
                        recall
                    )
                    if recall
                    is not None
                    else None
                ),

            "weighted_f1":
                (
                    float(
                        f1
                    )
                    if f1
                    is not None
                    else None
                ),
        },

        "output_directory":
            str(OUTPUT_DIR),
    }

    save_json(
        OUTPUT_DIR
        / "complete_report.json",
        master_report,
    )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print(
        "\n\n" + "=" * 80
    )

    print(
        "REPORT GENERATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nImages saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nGenerated PNG reports:"
    )

    for path in sorted(
        OUTPUT_DIR.glob(
            "*.png"
        )
    ):

        print(
            "  ",
            path.name
        )

    print(
        "\nAdditional files:"
    )

    print(
        "  complete_report.json"
    )

    if (
        OUTPUT_DIR
        / "test_predictions.csv"
    ).exists():

        print(
            "  test_predictions.csv"
        )

    if accuracy is not None:

        print(
            "\nFINAL TFLITE TEST ACCURACY:"
        )

        print(
            f"{accuracy * 100:.2f}%"
        )

    print(
        "\nH5 status:"
    )

    print(
        h5_info["load_status"]
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The H5 loading error does NOT prevent the TFLite"
    )

    print(
        "deployment model from being tested."
    )

    print(
        "This script evaluates the actual TFLite model"
    )

    print(
        "using the same MFCC preprocessing as your training code."
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()
