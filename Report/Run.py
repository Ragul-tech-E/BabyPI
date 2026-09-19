# ============================================================
# BABY CRY AI - COMPLETE MODEL & DATA ANALYSIS TOOL
# ============================================================
#
# This program:
#
# 1. Automatically finds H5/Keras models
# 2. Automatically finds TFLite model
# 3. Calculates model parameters
# 4. Calculates trainable/non-trainable parameters
# 5. Reports every layer
# 6. Reads label_map.json
# 7. Reads normalization.json
# 8. Reads deployment_info.json
# 9. Reads training_history.json
# 10. Reads training_log.csv
# 11. Searches for test data automatically
# 12. Evaluates test data if compatible data is found
# 13. Generates classification report
# 14. Generates confusion matrix
# 15. Generates normalized confusion matrix
# 16. Generates confidence distribution
# 17. Generates prediction distribution
# 18. Generates per-class metrics
# 19. Generates training curves
# 20. Generates model architecture report
# 21. Generates model parameter distribution
# 22. Generates model memory estimate
# 23. Generates TFLite information
# 24. Generates a complete technical report
# 25. Saves EVERYTHING as PNG images
#
# IMPORTANT:
# The program never invents test accuracy.
# If test data is not found, it clearly reports that.
#
# ============================================================

import os
import json
import glob
import traceback
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

# This script should be placed INSIDE your models folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "AI_REPORTS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Your actual class names.
# If label_map.json exists, the program will try to use it.
DEFAULT_CLASS_NAMES = [
    "Asphyxia",
    "Deaf",
    "Hunger",
    "Normal",
    "Pain"
]


# ============================================================
# DISPLAY SETTINGS
# ============================================================

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 10


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def title(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def safe_filename(text):
    return "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in str(text)
    )


def save_figure(filename):
    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print("Saved:", path)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def format_params(value):
    return f"{int(value):,}"


# ============================================================
# FIND FILES
# ============================================================

def find_all_files():

    files = []

    for root, dirs, filenames in os.walk(BASE_DIR):

        for filename in filenames:

            full_path = os.path.join(
                root,
                filename
            )

            files.append(full_path)

    return files


ALL_FILES = find_all_files()


def find_files_by_extension(extensions):

    result = []

    for file in ALL_FILES:

        if file.lower().endswith(
            tuple(extensions)
        ):
            result.append(file)

    return result


# ============================================================
# FIND MODELS
# ============================================================

title("SEARCHING FOR MODELS")

H5_MODELS = find_files_by_extension([
    ".h5",
    ".keras"
])

TFLITE_MODELS = find_files_by_extension([
    ".tflite"
])

print("\nH5/Keras models found:")

for f in H5_MODELS:
    print(" ", f)

print("\nTFLite models found:")

for f in TFLITE_MODELS:
    print(" ", f)


if not H5_MODELS:

    print("\nERROR:")
    print("No .h5 or .keras TensorFlow model was found.")

    print("\nModels folder:")
    print(BASE_DIR)

    raise SystemExit


# ============================================================
# CHOOSE BEST MODEL
# ============================================================

def model_priority(path):

    name = os.path.basename(path).lower()

    score = 0

    if "best" in name:
        score += 100

    if "final" in name:
        score += 80

    if "strong" in name:
        score += 20

    return score


H5_MODELS.sort(
    key=model_priority,
    reverse=True
)

MODEL_PATH = H5_MODELS[0]

title("SELECTED MODEL")

print("Selected:")
print(MODEL_PATH)

print(
    f"Size: {file_size_mb(MODEL_PATH):.3f} MB"
)


# ============================================================
# LOAD MODEL
# ============================================================

title("LOADING TENSORFLOW MODEL")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

except Exception as e:

    print("\nCould not load model:")
    print(e)

    traceback.print_exc()

    raise SystemExit


# ============================================================
# BASIC MODEL INFORMATION
# ============================================================

title("BASIC MODEL INFORMATION")

model_size = file_size_mb(
    MODEL_PATH
)

input_shape = model.input_shape
output_shape = model.output_shape

print("Model:")
print(os.path.basename(MODEL_PATH))

print("\nPath:")
print(MODEL_PATH)

print("\nModel size:")
print(f"{model_size:.3f} MB")

print("\nTensorFlow:")
print(tf.__version__)

print("\nInput shape:")
print(input_shape)

print("\nOutput shape:")
print(output_shape)


# ============================================================
# PARAMETER CALCULATION
# ============================================================

title("PARAMETER ANALYSIS")

total_params = model.count_params()

trainable_params = sum(
    int(np.prod(v.shape))
    for v in model.trainable_variables
)

non_trainable_params = (
    total_params -
    trainable_params
)

print("Total parameters:")
print(format_params(total_params))

print("\nTrainable parameters:")
print(format_params(trainable_params))

print("\nNon-trainable parameters:")
print(format_params(non_trainable_params))

print("\nMillion parameters:")
print(
    f"{total_params / 1_000_000:.4f} M"
)


# ============================================================
# MODEL PARAMETER REPORT IMAGE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.axis("off")

parameter_text = f"""
BABY CRY AI
MODEL PARAMETER ANALYSIS

Model
{os.path.basename(MODEL_PATH)}

TensorFlow
{tf.__version__}

Model Size
{model_size:.3f} MB

Input Shape
{input_shape}

Output Shape
{output_shape}

Total Parameters
{total_params:,}

Trainable Parameters
{trainable_params:,}

Non-Trainable Parameters
{non_trainable_params:,}

Total Parameters
{total_params / 1_000_000:.4f} Million
"""

plt.text(
    0.05,
    0.95,
    parameter_text,
    verticalalignment="top",
    fontsize=15,
    family="monospace"
)

plt.title(
    "AI Model Parameter Report",
    fontsize=18,
    fontweight="bold"
)

save_figure(
    "01_model_parameter_report.png"
)


# ============================================================
# LAYER REPORT
# ============================================================

title("LAYER-BY-LAYER ANALYSIS")

layer_rows = []

for i, layer in enumerate(model.layers):

    try:
        params = layer.count_params()
    except:
        params = 0

    try:
        output_shape_layer = str(
            layer.output_shape
        )
    except:
        output_shape_layer = "N/A"

    layer_rows.append({

        "Layer": i,

        "Name": layer.name,

        "Type": layer.__class__.__name__,

        "Parameters": params,

        "Output Shape":
            output_shape_layer

    })

layer_df = pd.DataFrame(
    layer_rows
)

print(layer_df.to_string(
    index=False
))


# ============================================================
# LAYER TABLE IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, max(5, len(layer_df) * 0.55 + 2))
)

ax.axis("off")

table = ax.table(
    cellText=layer_df.values,
    colLabels=layer_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(
    1,
    1.5
)

plt.title(
    "TensorFlow Model Architecture - Layer Details",
    fontsize=16,
    fontweight="bold"
)

save_figure(
    "02_layer_by_layer_report.png"
)


# ============================================================
# PARAMETER DISTRIBUTION
# ============================================================

if len(layer_df) > 0:

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        layer_df["Name"].astype(str),
        layer_df["Parameters"]
    )

    plt.xticks(
        rotation=60,
        ha="right"
    )

    plt.ylabel(
        "Number of Parameters"
    )

    plt.xlabel(
        "Layer"
    )

    plt.title(
        "Parameter Distribution Across Layers",
        fontweight="bold"
    )

    save_figure(
        "03_parameter_distribution.png"
    )


# ============================================================
# CLASS INFORMATION
# ============================================================

title("CLASS INFORMATION")

LABEL_MAP_PATH = os.path.join(
    BASE_DIR,
    "label_map.json"
)

class_names = DEFAULT_CLASS_NAMES.copy()

if os.path.exists(LABEL_MAP_PATH):

    try:

        with open(
            LABEL_MAP_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            label_data = json.load(f)

        print(
            "label_map.json found."
        )

        print(
            json.dumps(
                label_data,
                indent=4
            )
        )

        if isinstance(
            label_data,
            dict
        ):

            # Handles {"0":"Asphyxia", ...}
            try:

                keys = sorted(
                    label_data.keys(),
                    key=lambda x: int(x)
                )

                class_names = [
                    label_data[k]
                    for k in keys
                ]

            except:

                # Handles {"Asphyxia":0, ...}
                try:

                    class_names = [
                        k
                        for k, v in sorted(
                            label_data.items(),
                            key=lambda item: item[1]
                        )
                    ]

                except:
                    pass

    except Exception as e:

        print(
            "Could not read label_map.json:",
            e
        )


number_of_classes = (
    output_shape[-1]
    if isinstance(output_shape, tuple)
    else None
)

print("\nNumber of output classes:")
print(number_of_classes)

print("\nClass names:")

for i, name in enumerate(class_names):

    print(
        f"{i}: {name}"
    )


# ============================================================
# CLASS REPORT IMAGE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.axis("off")

class_text = (
    "BABY CRY AI\n\n"
    "CLASSIFICATION OUTPUTS\n\n"
)

class_text += (
    f"Number of output classes: "
    f"{number_of_classes}\n\n"
)

for i, name in enumerate(class_names):

    class_text += (
        f"{i}  ->  {name}\n"
    )

plt.text(
    0.05,
    0.95,
    class_text,
    verticalalignment="top",
    fontsize=16,
    family="monospace"
)

plt.title(
    "Class Mapping",
    fontsize=18,
    fontweight="bold"
)

save_figure(
    "04_class_mapping.png"
)


# ============================================================
# NORMALIZATION INFORMATION
# ============================================================

title("NORMALIZATION INFORMATION")

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "normalization.json"
)

normalization_data = None

if os.path.exists(
    NORMALIZATION_PATH
):

    try:

        with open(
            NORMALIZATION_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            normalization_data = json.load(f)

        print(
            json.dumps(
                normalization_data,
                indent=4
            )
        )

    except Exception as e:

        print(
            "Could not read normalization.json:",
            e
        )

else:

    print(
        "normalization.json not found."
    )


# ============================================================
# DEPLOYMENT INFORMATION
# ============================================================

title("DEPLOYMENT INFORMATION")

DEPLOYMENT_PATH = os.path.join(
    BASE_DIR,
    "deployment_info.json"
)

deployment_data = None

if os.path.exists(
    DEPLOYMENT_PATH
):

    try:

        with open(
            DEPLOYMENT_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            deployment_data = json.load(f)

        print(
            json.dumps(
                deployment_data,
                indent=4
            )
        )

    except Exception as e:

        print(
            "Could not read deployment_info.json:",
            e
        )

else:

    print(
        "deployment_info.json not found."
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

title("TRAINING HISTORY")

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "training_history.json"
)

history_data = None

if os.path.exists(
    HISTORY_PATH
):

    try:

        with open(
            HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            history_data = json.load(f)

        print(
            "training_history.json loaded."
        )

    except Exception as e:

        print(
            "Could not load training history:",
            e
        )


# ============================================================
# TRAINING HISTORY IMAGE
# ============================================================

if history_data is not None:

    if isinstance(
        history_data,
        dict
    ):

        # Sometimes stored as {"history": {...}}
        if (
            "history" in history_data
            and isinstance(
                history_data["history"],
                dict
            )
        ):

            history = history_data[
                "history"
            ]

        else:

            history = history_data

        keys = list(
            history.keys()
        )

        print(
            "History keys:",
            keys
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy_keys = [
            k for k in keys
            if k.lower() in [
                "accuracy",
                "acc",
                "categorical_accuracy"
            ]
        ]

        validation_accuracy_keys = [
            k for k in keys
            if (
                "val_" in k.lower()
                and (
                    "accuracy" in k.lower()
                    or "acc" in k.lower()
                )
            )
        ]

        if accuracy_keys:

            plt.figure(
                figsize=(11, 6)
            )

            plt.plot(
                history[accuracy_keys[0]],
                label="Training Accuracy"
            )

            if validation_accuracy_keys:

                plt.plot(
                    history[
                        validation_accuracy_keys[0]
                    ],
                    label="Validation Accuracy"
                )

            plt.xlabel(
                "Epoch"
            )

            plt.ylabel(
                "Accuracy"
            )

            plt.title(
                "Training and Validation Accuracy",
                fontweight="bold"
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            save_figure(
                "05_training_accuracy.png"
            )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss_keys = [
            k for k in keys
            if k.lower() == "loss"
        ]

        validation_loss_keys = [
            k for k in keys
            if k.lower() == "val_loss"
        ]

        if loss_keys:

            plt.figure(
                figsize=(11, 6)
            )

            plt.plot(
                history[loss_keys[0]],
                label="Training Loss"
            )

            if validation_loss_keys:

                plt.plot(
                    history[
                        validation_loss_keys[0]
                    ],
                    label="Validation Loss"
                )

            plt.xlabel(
                "Epoch"
            )

            plt.ylabel(
                "Loss"
            )

            plt.title(
                "Training and Validation Loss",
                fontweight="bold"
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            save_figure(
                "06_training_loss.png"
            )


# ============================================================
# TRAINING CSV
# ============================================================

title("TRAINING LOG")

CSV_FILES = find_files_by_extension([
    ".csv"
])

print(
    "CSV files found:"
)

for f in CSV_FILES:
    print(
        " ",
        f
    )


training_csv = None

for f in CSV_FILES:

    if "training" in os.path.basename(
        f
    ).lower():

        training_csv = f
        break


if training_csv:

    try:

        train_df = pd.read_csv(
            training_csv
        )

        print(
            train_df.head()
        )

        print(
            "\nColumns:"
        )

        print(
            train_df.columns.tolist()
        )

    except Exception as e:

        print(
            "Could not read training CSV:",
            e
        )


# ============================================================
# SEARCH FOR TEST DATA
# ============================================================

title("SEARCHING FOR TEST DATA")

NPY_FILES = find_files_by_extension([
    ".npy",
    ".npz"
])

print(
    "\nNumPy data files:"
)

for f in NPY_FILES:

    print(
        " ",
        f
    )


# ============================================================
# AUTOMATIC TEST DATA DETECTION
# ============================================================

X_TEST = None
Y_TEST = None

X_TEST_PATH = None
Y_TEST_PATH = None


def filename_score_x(path):

    name = os.path.basename(
        path
    ).lower()

    score = 0

    if "x_test" in name:
        score += 100

    if "test_x" in name:
        score += 90

    if "features_test" in name:
        score += 80

    if "test_features" in name:
        score += 70

    if "test" in name:
        score += 20

    if "feature" in name:
        score += 10

    return score


def filename_score_y(path):

    name = os.path.basename(
        path
    ).lower()

    score = 0

    if "y_test" in name:
        score += 100

    if "test_y" in name:
        score += 90

    if "labels_test" in name:
        score += 80

    if "test_labels" in name:
        score += 70

    if "label" in name:
        score += 20

    if "test" in name:
        score += 20

    return score


x_candidates = sorted(
    NPY_FILES,
    key=filename_score_x,
    reverse=True
)

y_candidates = sorted(
    NPY_FILES,
    key=filename_score_y,
    reverse=True
)


for x_path in x_candidates:

    if filename_score_x(x_path) <= 0:
        continue

    for y_path in y_candidates:

        if filename_score_y(y_path) <= 0:
            continue

        if x_path == y_path:
            continue

        X_TEST_PATH = x_path
        Y_TEST_PATH = y_path

        break

    if X_TEST_PATH:
        break


# ============================================================
# TRY NPZ FILES
# ============================================================

if X_TEST_PATH is None:

    for path in NPY_FILES:

        if not path.endswith(
            ".npz"
        ):
            continue

        try:

            data = np.load(
                path,
                allow_pickle=True
            )

            keys = list(
                data.keys()
            )

            print(
                "\nNPZ:",
                path
            )

            print(
                "Keys:",
                keys
            )

            x_key = None
            y_key = None

            for key in keys:

                kl = key.lower()

                if (
                    kl in ["x_test", "test_x"]
                    or "test_feature" in kl
                ):

                    x_key = key

                if (
                    kl in ["y_test", "test_y"]
                    or "test_label" in kl
                ):

                    y_key = key

            if x_key and y_key:

                X_TEST = data[x_key]
                Y_TEST = data[y_key]

                X_TEST_PATH = path
                Y_TEST_PATH = path

                break

        except Exception as e:

            print(
                "NPZ error:",
                e
            )


# ============================================================
# LOAD TEST DATA
# ============================================================

if X_TEST_PATH and Y_TEST_PATH:

    try:

        if X_TEST is None:

            X_TEST = np.load(
                X_TEST_PATH,
                allow_pickle=True
            )

        if Y_TEST is None:

            Y_TEST = np.load(
                Y_TEST_PATH,
                allow_pickle=True
            )

        print(
            "\nTEST DATA FOUND!"
        )

        print(
            "X:",
            X_TEST_PATH
        )

        print(
            "Y:",
            Y_TEST_PATH
        )

        print(
            "X shape:",
            X_TEST.shape
        )

        print(
            "Y shape:",
            Y_TEST.shape
        )

    except Exception as e:

        print(
            "\nCould not load test data:"
        )

        print(e)

        X_TEST = None
        Y_TEST = None


# ============================================================
# TEST DATA REPORT
# ============================================================

if X_TEST is not None and Y_TEST is not None:

    title("TEST DATA ANALYSIS")

    print(
        "Number of test samples:",
        len(X_TEST)
    )

    print(
        "Test feature shape:",
        X_TEST.shape
    )

    print(
        "Test label shape:",
        Y_TEST.shape
    )

    # --------------------------------------------------------
    # Convert labels
    # --------------------------------------------------------

    if Y_TEST.ndim > 1:

        if Y_TEST.shape[-1] > 1:

            y_true = np.argmax(
                Y_TEST,
                axis=1
            )

        else:

            y_true = Y_TEST.reshape(-1)

    else:

        y_true = Y_TEST.reshape(-1)


    # --------------------------------------------------------
    # Test dataset overview image
    # --------------------------------------------------------

    unique_labels, counts = np.unique(
        y_true,
        return_counts=True
    )

    label_names = []

    for label in unique_labels:

        try:

            label_int = int(label)

            if (
                0 <= label_int
                < len(class_names)
            ):

                label_names.append(
                    class_names[label_int]
                )

            else:

                label_names.append(
                    str(label)
                )

        except:

            label_names.append(
                str(label)
            )


    plt.figure(
        figsize=(11, 6)
    )

    bars = plt.bar(
        label_names,
        counts
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of Test Samples"
    )

    plt.title(
        "Test Dataset Class Distribution",
        fontweight="bold"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    for bar, count in zip(
        bars,
        counts
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom"
        )

    save_figure(
        "07_test_dataset_distribution.png"
    )


    # ========================================================
    # TEST INFERENCE
    # ========================================================

    title("RUNNING TEST INFERENCE")

    try:

        predictions = model.predict(
            X_TEST,
            verbose=1
        )

    except Exception as e:

        print(
            "\nMODEL / TEST DATA SHAPE MISMATCH"
        )

        print(
            "This means the saved test features do not have"
        )

        print(
            "the exact input format expected by the model."
        )

        print(
            "\nModel input:",
            model.input_shape
        )

        print(
            "Test input:",
            X_TEST.shape
        )

        print(
            "\nError:"
        )

        print(e)

        predictions = None


    # ========================================================
    # METRICS
    # ========================================================

    if predictions is not None:

        if predictions.ndim == 2:

            y_pred = np.argmax(
                predictions,
                axis=1
            )

            confidence = np.max(
                predictions,
                axis=1
            )

        else:

            y_pred = (
                predictions > 0.5
            ).astype(int).reshape(-1)

            confidence = np.maximum(
                predictions.reshape(-1),
                1 - predictions.reshape(-1)
            )


        # ----------------------------------------------------
        # Basic metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )


        print(
            "\nAccuracy:",
            f"{accuracy * 100:.2f}%"
        )

        print(
            "Precision:",
            f"{precision * 100:.2f}%"
        )

        print(
            "Recall:",
            f"{recall * 100:.2f}%"
        )

        print(
            "F1:",
            f"{f1 * 100:.2f}%"
        )


        # ====================================================
        # PERFORMANCE SUMMARY IMAGE
        # ====================================================

        plt.figure(
            figsize=(11, 7)
        )

        plt.axis("off")

        performance_text = f"""
BABY CRY AI
TEST DATA PERFORMANCE

Test Samples
{len(y_true):,}

Accuracy
{accuracy * 100:.2f}%

Weighted Precision
{precision * 100:.2f}%

Weighted Recall
{recall * 100:.2f}%

Weighted F1-Score
{f1 * 100:.2f}%

Average Prediction Confidence
{np.mean(confidence) * 100:.2f}%

Minimum Prediction Confidence
{np.min(confidence) * 100:.2f}%

Maximum Prediction Confidence
{np.max(confidence) * 100:.2f}%
"""

        plt.text(
            0.05,
            0.95,
            performance_text,
            verticalalignment="top",
            fontsize=17,
            family="monospace"
        )

        plt.title(
            "Test Dataset Performance",
            fontsize=19,
            fontweight="bold"
        )

        save_figure(
            "08_test_performance_summary.png"
        )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names
            if len(class_names)
            == number_of_classes
            else None,
            zero_division=0,
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        print(
            "\nClassification Report:"
        )

        print(
            report_df
        )


        # ----------------------------------------------------
        # Classification report image
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        ax.axis("off")

        display_df = report_df.round(4)

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            rowLabels=display_df.index,
            loc="center",
            cellLoc="center"
        )

        table.auto_set_font_size(
            False
        )

        table.set_fontsize(
            10
        )

        table.scale(
            1,
            1.5
        )

        plt.title(
            "Detailed Classification Report",
            fontsize=17,
            fontweight="bold"
        )

        save_figure(
            "09_classification_report.png"
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        labels_for_cm = (
            class_names
            if len(class_names)
            == number_of_classes
            else [
                str(i)
                for i in range(
                    number_of_classes
                )
            ]
        )

        plt.figure(
            figsize=(9, 7)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels_for_cm,
            yticklabels=labels_for_cm
        )

        plt.xlabel(
            "Predicted Class"
        )

        plt.ylabel(
            "Actual Class"
        )

        plt.title(
            "Confusion Matrix",
            fontweight="bold"
        )

        save_figure(
            "10_confusion_matrix.png"
        )


        # ====================================================
        # NORMALIZED CONFUSION MATRIX
        # ====================================================

        cm_normalized = (
            cm.astype(float)
            /
            np.maximum(
                cm.sum(axis=1, keepdims=True),
                1
            )
        )

        plt.figure(
            figsize=(9, 7)
        )

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels_for_cm,
            yticklabels=labels_for_cm,
            vmin=0,
            vmax=1
        )

        plt.xlabel(
            "Predicted Class"
        )

        plt.ylabel(
            "Actual Class"
        )

        plt.title(
            "Normalized Confusion Matrix",
            fontweight="bold"
        )

        save_figure(
            "11_normalized_confusion_matrix.png"
        )


        # ====================================================
        # CONFIDENCE DISTRIBUTION
        # ====================================================

        plt.figure(
            figsize=(11, 6)
        )

        plt.hist(
            confidence * 100,
            bins=20,
            edgecolor="black"
        )

        plt.xlabel(
            "Prediction Confidence (%)"
        )

        plt.ylabel(
            "Number of Samples"
        )

        plt.title(
            "Model Prediction Confidence Distribution",
            fontweight="bold"
        )

        save_figure(
            "12_prediction_confidence.png"
        )


        # ====================================================
        # CORRECT VS INCORRECT
        # ====================================================

        correct = (
            y_true == y_pred
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.bar(
            ["Correct", "Incorrect"],
            [
                np.sum(correct),
                np.sum(~correct)
            ]
        )

        plt.ylabel(
            "Number of Samples"
        )

        plt.title(
            "Correct vs Incorrect Predictions",
            fontweight="bold"
        )

        save_figure(
            "13_correct_incorrect_predictions.png"
        )


        # ====================================================
        # PER-CLASS ACCURACY / RECALL
        # ====================================================

        per_class_recall = []

        for i in range(
            number_of_classes
        ):

            actual = (
                y_true == i
            )

            if np.sum(actual) > 0:

                value = np.sum(
                    (
                        y_pred[actual]
                        == i
                    )
                ) / np.sum(actual)

            else:

                value = 0

            per_class_recall.append(
                value
            )


        plt.figure(
            figsize=(11, 6)
        )

        bars = plt.bar(
            labels_for_cm,
            np.array(
                per_class_recall
            ) * 100
        )

        plt.ylabel(
            "Recall (%)"
        )

        plt.xlabel(
            "Class"
        )

        plt.title(
            "Per-Class Recall",
            fontweight="bold"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.ylim(
            0,
            100
        )

        for bar, value in zip(
            bars,
            per_class_recall
        ):

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value * 100,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom"
            )

        save_figure(
            "14_per_class_recall.png"
        )


        # ====================================================
        # SAMPLE PREDICTION TABLE
        # ====================================================

        sample_count = min(
            len(y_true),
            100
        )

        sample_rows = []

        for i in range(
            sample_count
        ):

            true_id = int(
                y_true[i]
            )

            pred_id = int(
                y_pred[i]
            )

            true_name = (
                labels_for_cm[true_id]
                if (
                    0 <= true_id
                    < len(labels_for_cm)
                )
                else str(true_id)
            )

            pred_name = (
                labels_for_cm[pred_id]
                if (
                    0 <= pred_id
                    < len(labels_for_cm)
                )
                else str(pred_id)
            )

            sample_rows.append([
                i,
                true_name,
                pred_name,
                f"{confidence[i] * 100:.2f}%",
                "YES"
                if true_id == pred_id
                else "NO"
            ])


        sample_df = pd.DataFrame(
            sample_rows,
            columns=[
                "Sample",
                "Actual",
                "Predicted",
                "Confidence",
                "Correct"
            ]
        )


        fig, ax = plt.subplots(
            figsize=(12, 18)
        )

        ax.axis("off")

        table = ax.table(
            cellText=sample_df.values,
            colLabels=sample_df.columns,
            loc="center",
            cellLoc="center"
        )

        table.auto_set_font_size(
            False
        )

        table.set_fontsize(
            8
        )

        table.scale(
            1,
            1.3
        )

        plt.title(
            f"First {sample_count} Test Predictions",
            fontsize=16,
            fontweight="bold"
        )

        save_figure(
            "15_test_prediction_samples.png"
        )


# ============================================================
# NO TEST DATA
# ============================================================

else:

    print(
        "\nNo compatible X_test / Y_test dataset was found."
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.axis("off")

    text = """
TEST DATA EVALUATION

No compatible test dataset was found automatically.

The program therefore DID NOT calculate
test accuracy or test performance.

This is intentional.

To calculate genuine test performance, provide:

X_test + y_test

in .npy or .npz format, using exactly
the preprocessing/input format expected
by the TensorFlow model.
"""

    plt.text(
        0.05,
        0.9,
        text,
        verticalalignment="top",
        fontsize=16,
        family="monospace"
    )

    plt.title(
        "Test Data Status",
        fontsize=18,
        fontweight="bold"
    )

    save_figure(
        "07_TEST_DATA_NOT_FOUND.png"
    )


# ============================================================
# TFLITE ANALYSIS
# ============================================================

title("TENSORFLOW LITE ANALYSIS")

if TFLITE_MODELS:

    for index, tflite_path in enumerate(
        TFLITE_MODELS
    ):

        print(
            "\nTFLite model:",
            tflite_path
        )

        try:

            interpreter = tf.lite.Interpreter(
                model_path=tflite_path
            )

            interpreter.allocate_tensors()

            input_details = (
                interpreter.get_input_details()
            )

            output_details = (
                interpreter.get_output_details()
            )

            print(
                "Input:",
                input_details
            )

            print(
                "Output:",
                output_details
            )


            # ------------------------------------------------
            # TFLite report
            # ------------------------------------------------

            report_text = f"""
TENSORFLOW LITE MODEL

File
{os.path.basename(tflite_path)}

Size
{file_size_mb(tflite_path):.3f} MB

Input Details
{json.dumps(
    input_details,
    default=str,
    indent=2
)}

Output Details
{json.dumps(
    output_details,
    default=str,
    indent=2
)}
"""

            plt.figure(
                figsize=(14, 10)
            )

            plt.axis("off")

            plt.text(
                0.02,
                0.98,
                report_text,
                verticalalignment="top",
                fontsize=9,
                family="monospace"
            )

            plt.title(
                "TensorFlow Lite Deployment Report",
                fontsize=17,
                fontweight="bold"
            )

            save_figure(
                f"16_tflite_report_{index + 1}.png"
            )


        except Exception as e:

            print(
                "TFLite analysis failed:",
                e
            )

else:

    print(
        "No TFLite model found."
    )


# ============================================================
# MODEL MEMORY ESTIMATION
# ============================================================

title("MODEL MEMORY ESTIMATION")

# FP32 = 4 bytes
# FP16 = 2 bytes
# INT8 = 1 byte

fp32_mb = (
    total_params * 4
    / (1024 * 1024)
)

fp16_mb = (
    total_params * 2
    / (1024 * 1024)
)

int8_mb = (
    total_params
    / (1024 * 1024)
)

memory_df = pd.DataFrame({

    "Format": [
        "FP32",
        "FP16",
        "INT8"
    ],

    "Bytes/Parameter": [
        4,
        2,
        1
    ],

    "Estimated Weight Memory (MB)": [
        fp32_mb,
        fp16_mb,
        int8_mb
    ]

})

print(
    memory_df.to_string(
        index=False
    )
)


# ============================================================
# MEMORY IMAGE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    memory_df["Format"],
    memory_df[
        "Estimated Weight Memory (MB)"
    ]
)

plt.ylabel(
    "Estimated Weight Memory (MB)"
)

plt.title(
    "Estimated Neural Network Weight Memory",
    fontweight="bold"
)

for bar, value in zip(
    bars,
    memory_df[
        "Estimated Weight Memory (MB)"
    ]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f} MB",
        ha="center",
        va="bottom"
    )

save_figure(
    "17_model_memory_estimation.png"
)


# ============================================================
# COMPLETE TECHNICAL REPORT
# ============================================================

title("CREATING COMPLETE TECHNICAL REPORT")

report_lines = []

report_lines.append(
    "BABY CRY AI - COMPLETE TECHNICAL REPORT"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    f"Model: {os.path.basename(MODEL_PATH)}"
)

report_lines.append(
    f"Model size: {model_size:.3f} MB"
)

report_lines.append(
    f"TensorFlow version: {tf.__version__}"
)

report_lines.append(
    f"Input shape: {input_shape}"
)

report_lines.append(
    f"Output shape: {output_shape}"
)

report_lines.append(
    f"Number of classes: {number_of_classes}"
)

report_lines.append(
    ""
)

report_lines.append(
    f"Total parameters: {total_params:,}"
)

report_lines.append(
    f"Trainable parameters: {trainable_params:,}"
)

report_lines.append(
    f"Non-trainable parameters: {non_trainable_params:,}"
)

report_lines.append(
    f"Parameters: {total_params / 1_000_000:.4f} M"
)

report_lines.append(
    ""
)

report_lines.append(
    "CLASS MAPPING"
)

for i, name in enumerate(
    class_names
):

    report_lines.append(
        f"{i}: {name}"
    )


if X_TEST is not None and Y_TEST is not None:

    if "accuracy" in locals():

        report_lines.append(
            ""
        )

        report_lines.append(
            "TEST PERFORMANCE"
        )

        report_lines.append(
            f"Test samples: {len(y_true):,}"
        )

        report_lines.append(
            f"Accuracy: {accuracy * 100:.2f}%"
        )

        report_lines.append(
            f"Precision: {precision * 100:.2f}%"
        )

        report_lines.append(
            f"Recall: {recall * 100:.2f}%"
        )

        report_lines.append(
            f"F1-score: {f1 * 100:.2f}%"
        )

        report_lines.append(
            f"Average confidence: "
            f"{np.mean(confidence) * 100:.2f}%"
        )

else:

    report_lines.append(
        ""
    )

    report_lines.append(
        "TEST DATA"
    )

    report_lines.append(
        "No compatible test dataset found."
    )


report_lines.append(
    ""
)

report_lines.append(
    "ESTIMATED MODEL WEIGHT MEMORY"
)

report_lines.append(
    f"FP32: {fp32_mb:.3f} MB"
)

report_lines.append(
    f"FP16: {fp16_mb:.3f} MB"
)

report_lines.append(
    f"INT8: {int8_mb:.3f} MB"
)


complete_report = "\n".join(
    report_lines
)

print(
    complete_report
)


# ============================================================
# COMPLETE REPORT IMAGE
# ============================================================

plt.figure(
    figsize=(14, 18)
)

plt.axis("off")

plt.text(
    0.03,
    0.98,
    complete_report,
    verticalalignment="top",
    fontsize=11,
    family="monospace"
)

plt.title(
    "Baby Cry AI - Complete Technical Report",
    fontsize=19,
    fontweight="bold"
)

save_figure(
    "18_COMPLETE_TECHNICAL_REPORT.png"
)


# ============================================================
# SAVE RAW REPORT TXT
# ============================================================

report_txt_path = os.path.join(
    OUTPUT_DIR,
    "complete_technical_report.txt"
)

with open(
    report_txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        complete_report
    )


# ============================================================
# SAVE JSON
# ============================================================

json_report = {

    "model": {
        "file": os.path.basename(
            MODEL_PATH
        ),

        "size_MB": model_size,

        "tensorflow_version":
            tf.__version__,

        "input_shape":
            str(input_shape),

        "output_shape":
            str(output_shape),

        "classes":
            class_names,

        "number_of_classes":
            number_of_classes,

        "total_parameters":
            int(total_params),

        "trainable_parameters":
            int(trainable_params),

        "non_trainable_parameters":
            int(non_trainable_params),

        "parameters_million":
            total_params / 1_000_000,

        "estimated_memory": {
            "FP32_MB": fp32_mb,
            "FP16_MB": fp16_mb,
            "INT8_MB": int8_mb
        }
    }

}


if X_TEST is not None and Y_TEST is not None:

    if "accuracy" in locals():

        json_report["test_performance"] = {

            "samples":
                int(len(y_true)),

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1),

            "average_confidence":
                float(np.mean(confidence))

        }


json_path = os.path.join(
    OUTPUT_DIR,
    "complete_model_report.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_report,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

title("ANALYSIS COMPLETE")

print(
    "\nALL REPORTS HAVE BEEN SAVED HERE:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGenerated files:"
)

for file in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        " ",
        file
    )

print(
    "\nIMPORTANT:"
)

print(
    "The program only reports test accuracy if"
)

print(
    "actual compatible test data was found."
)

print(
    "\nDone."
)