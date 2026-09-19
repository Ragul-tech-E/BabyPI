# ============================================================
# BABY CRY AI - COMPLETE MODEL REPORT GENERATOR
#
# Works primarily from TFLITE.
# H5 is optional because custom FixedStandardization may prevent
# direct H5 loading.
#
# Generates:
#   - Console report
#   - Complete model statistics
#   - Parameter count
#   - Parameters in millions
#   - Layer count
#   - TFLite operator/node count
#   - Activation neuron statistics
#   - Test accuracy
#   - Precision / Recall / F1
#   - Classification report
#   - Confusion matrix
#   - Class distribution
#   - Model architecture image
#   - Training graphs
#   - JSON report
#   - Text report
#
# ============================================================

import os
import json
import math
import numpy as np

# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

try:
    import tensorflow as tf
except Exception as e:
    print("TensorFlow import failed:", e)
    raise

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )
except Exception:
    accuracy_score = None


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ------------------------------------------------------------
# Your actual model
# ------------------------------------------------------------

TFLITE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "baby_audio_strong.tflite"
)

H5_MODEL_PATH = os.path.join(
    BASE_DIR,
    "baby_audio_strong_best.h5"
)

# ------------------------------------------------------------
# Other files from your training
# ------------------------------------------------------------

LABEL_MAP_PATH = os.path.join(
    BASE_DIR,
    "label_map.json"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "normalization.json"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "training_history.json"
)

CLASSIFICATION_REPORT_PATH = os.path.join(
    BASE_DIR,
    "classification_report.txt"
)

TRAINING_LOG_PATH = os.path.join(
    BASE_DIR,
    "training_log.csv"
)

# ------------------------------------------------------------
# Optional test arrays
#
# If these exist, test evaluation will be performed.
# ------------------------------------------------------------

X_TEST_PATH = os.path.join(
    BASE_DIR,
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR,
    "y_test.npy"
)

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

REPORT_DIR = os.path.join(
    BASE_DIR,
    "AI_REPORT"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# DEFAULT CLASS NAMES
# ============================================================

DEFAULT_CLASS_NAMES = [
    "Asphyxia",
    "Deaf",
    "Hunger",
    "Normal",
    "Pain"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def separator():
    return "=" * 90


def format_number(value):
    return f"{int(value):,}"


def format_millions(value):
    return f"{value / 1_000_000:.4f} M"


def format_kb(size):
    return f"{size / 1024:.2f} KB"


def format_mb(size):
    return f"{size / 1024 / 1024:.3f} MB"


def safe_shape(shape):
    result = []

    for x in shape:
        try:
            result.append(int(x))
        except Exception:
            result.append(-1)

    return result


def product_known_dimensions(shape):
    result = 1

    for dim in shape:

        try:
            dim = int(dim)
        except Exception:
            return None

        if dim <= 0:
            return None

        result *= dim

    return result


def print_section(title):

    print()
    print(separator())
    print(title)
    print(separator())


# ============================================================
# FIND MODEL
# ============================================================

print(separator())
print("BABY CRY AI MODEL REPORT GENERATOR")
print(separator())

print()
print("Base folder:")
print(BASE_DIR)

print()
print("TFLite:")
print(TFLITE_MODEL_PATH)

print()
print("H5:")
print(H5_MODEL_PATH)


if not os.path.exists(TFLITE_MODEL_PATH):

    print()
    print("ERROR:")
    print("TFLite model not found:")
    print(TFLITE_MODEL_PATH)

    raise SystemExit


# ============================================================
# LOAD LABEL MAP
# ============================================================

CLASS_NAMES = DEFAULT_CLASS_NAMES.copy()

if os.path.exists(LABEL_MAP_PATH):

    try:

        with open(
            LABEL_MAP_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            label_map = json.load(f)

        ordered = []

        for key in sorted(
            label_map.keys(),
            key=lambda x: int(x)
        ):

            ordered.append(
                label_map[key]
            )

        if len(ordered) > 0:
            CLASS_NAMES = ordered

    except Exception as e:

        print(
            "WARNING: Could not read label_map.json:",
            e
        )


# ============================================================
# FILE INFORMATION
# ============================================================

tflite_size = os.path.getsize(
    TFLITE_MODEL_PATH
)

h5_size = None

if os.path.exists(H5_MODEL_PATH):

    h5_size = os.path.getsize(
        H5_MODEL_PATH
    )


print_section(
    "MODEL FILE INFORMATION"
)

print(
    "TFLite file:",
    os.path.basename(
        TFLITE_MODEL_PATH
    )
)

print(
    "TFLite size:",
    format_mb(tflite_size)
)

if h5_size is not None:

    print(
        "H5 file:",
        os.path.basename(
            H5_MODEL_PATH
        )
    )

    print(
        "H5 size:",
        format_mb(h5_size)
    )


# ============================================================
# LOAD TFLITE
# ============================================================

print_section(
    "LOADING TFLITE MODEL"
)

try:

    interpreter = tf.lite.Interpreter(
        model_path=TFLITE_MODEL_PATH,
        num_threads=4
    )

    interpreter.allocate_tensors()

    print(
        "TFLite model loaded successfully."
    )

except Exception as e:

    print()
    print("FAILED TO LOAD TFLITE MODEL")
    print(e)

    raise SystemExit


# ============================================================
# TENSOR INFORMATION
# ============================================================

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)

tensor_details = (
    interpreter.get_tensor_details()
)


# ============================================================
# INPUT INFORMATION
# ============================================================

print_section(
    "INPUT / OUTPUT INFORMATION"
)

input_detail = input_details[0]

output_detail = output_details[0]

input_shape = safe_shape(
    input_detail["shape"]
)

output_shape = safe_shape(
    output_detail["shape"]
)

print()
print("Input name:")
print(input_detail["name"])

print()
print("Input shape:")
print(input_shape)

print()
print("Input dtype:")
print(input_detail["dtype"])

print()
print("Output name:")
print(output_detail["name"])

print()
print("Output shape:")
print(output_shape)

print()
print("Output dtype:")
print(output_detail["dtype"])

print()
print("Number of output classes:")

if len(output_shape) > 0:

    num_classes = output_shape[-1]

else:

    num_classes = len(CLASS_NAMES)

print(num_classes)


# ============================================================
# MODEL PARAMETERS FROM TFLITE CONSTANT TENSORS
# ============================================================

print_section(
    "PARAMETER ANALYSIS"
)

parameter_tensors = []

total_parameters = 0

for tensor in tensor_details:

    shape = safe_shape(
        tensor["shape"]
    )

    # Constant tensors normally have buffer data.
    #
    # We identify weight-like tensors by tensor names.
    # This avoids counting ordinary activation tensors.

    name = tensor["name"].lower()

    is_parameter = any(
        keyword in name
        for keyword in [
            "kernel",
            "weight",
            "bias",
            "depthwise",
            "pointwise",
            "gamma",
            "beta",
            "moving_mean",
            "moving_variance",
            "mean",
            "std"
        ]
    )

    count = product_known_dimensions(
        shape
    )

    if (
        is_parameter
        and count is not None
        and count > 0
    ):

        parameter_tensors.append(
            {
                "name": tensor["name"],
                "shape": shape,
                "parameters": count
            }
        )

        total_parameters += count


print()
print("Parameter tensors found:")
print(
    len(parameter_tensors)
)

print()
print("Estimated TFLite parameters:")
print(
    format_number(
        total_parameters
    )
)

print()
print("Parameters in millions:")
print(
    format_millions(
        total_parameters
    )


)

# ============================================================
# IMPROVED PARAMETER COUNT
# ============================================================

# Some TensorFlow Lite versions rename tensors.
# Therefore also inspect all tensors with constant buffers.

try:

    all_constant_parameters = 0

    constant_tensor_list = []

    for tensor in tensor_details:

        shape = safe_shape(
            tensor["shape"]
        )

        count = product_known_dimensions(
            shape
        )

        if count is None:
            continue

        tensor_name = tensor["name"]

        # TensorFlow Lite exposes buffer index.
        # Weight tensors generally have non-zero buffers.

        buffer_index = tensor.get(
            "buffer",
            0
        )

        if (
            buffer_index is not None
            and int(buffer_index) != 0
            and count > 0
        ):

            # Ignore very large activation-like tensors.
            #
            # Parameter names are preferred.
            name_lower = tensor_name.lower()

            if any(
                k in name_lower
                for k in [
                    "kernel",
                    "weight",
                    "bias",
                    "depthwise",
                    "pointwise",
                    "gamma",
                    "beta",
                    "mean",
                    "variance",
                    "std"
                ]
            ):

                constant_tensor_list.append(
                    tensor_name
                )

    print()
    print(
        "Weight-like constant tensors:"
    )

    print(
        len(
            constant_tensor_list
        )
    )

except Exception as e:

    print(
        "Parameter secondary analysis:",
        e
    )


# ============================================================
# TFLITE OPERATOR / NODE COUNT
# ============================================================

print_section(
    "TFLITE NODE / OPERATOR ANALYSIS"
)

try:

    # Private API gives actual operator details
    ops = interpreter._get_ops_details()

    node_count = len(ops)

except Exception:

    ops = []

    node_count = 0


print()
print("TFLite operator/node count:")
print(
    format_number(
        node_count
    )
)


# ============================================================
# OPERATOR DISTRIBUTION
# ============================================================

operator_counts = {}

for op in ops:

    op_name = op.get(
        "op_name",
        "UNKNOWN"
    )

    operator_counts[op_name] = (
        operator_counts.get(
            op_name,
            0
        ) + 1
    )


print()
print("Operator distribution:")

for name, count in sorted(
    operator_counts.items(),
    key=lambda x: x[0]
):

    print(
        f"  {name:<35} {count}"
    )


# ============================================================
# ACTIVATION / NEURON ANALYSIS
# ============================================================

print_section(
    "ACTIVATION / NEURON ANALYSIS"
)

activation_tensors = []

total_activation_neurons = 0

max_activation_neurons = 0

max_activation_tensor = None

for tensor in tensor_details:

    shape = safe_shape(
        tensor["shape"]
    )

    count = product_known_dimensions(
        shape
    )

    if count is None:
        continue

    # Ignore scalar constants.
    if count <= 1:
        continue

    activation_tensors.append(
        {
            "name": tensor["name"],
            "shape": shape,
            "neurons": count
        }
    )

    total_activation_neurons += count

    if count > max_activation_neurons:

        max_activation_neurons = count

        max_activation_tensor = tensor


print()
print(
    "Activation tensors:"
)

print(
    format_number(
        len(activation_tensors)
    )
)

print()
print(
    "Total activation values:"
)

print(
    format_number(
        total_activation_neurons
    )
)

print()
print(
    "Total activation values in millions:"
)

print(
    format_millions(
        total_activation_neurons
    )
)

print()
print(
    "Maximum neurons in one tensor:"
)

print(
    format_number(
        max_activation_neurons
    )
)

if max_activation_tensor:

    print()
    print(
        "Largest activation tensor:"
    )

    print(
        max_activation_tensor["name"]
    )

    print()
    print(
        "Shape:"
    )

    print(
        safe_shape(
            max_activation_tensor["shape"]
        )
    )


# ============================================================
# CLASS INFORMATION
# ============================================================

print_section(
    "CLASS INFORMATION"
)

print()
print(
    "Number of classes:"
)

print(
    num_classes
)

print()
print(
    "Class mapping:"
)

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"  {i} -> {name}"
    )


# ============================================================
# H5 PARAMETER ANALYSIS
# ============================================================

h5_parameter_count = None

try:

    import h5py

    if os.path.exists(
        H5_MODEL_PATH
    ):

        print_section(
            "H5 WEIGHT ANALYSIS"
        )

        h5_total = 0

        with h5py.File(
            H5_MODEL_PATH,
            "r"
        ) as h5:

            def visit_dataset(name, obj):

                nonlocal_dummy = None

                global h5_total

                if isinstance(
                    obj,
                    h5py.Dataset
                ):

                    shape = obj.shape

                    count = 1

                    for dim in shape:
                        count *= dim

                    # Ignore optimizer variables.
                    if "optimizer" not in name.lower():

                        h5_total_local[0] += count

            h5_total_local = [0]

            h5.visititems(
                visit_dataset
            )

            h5_parameter_count = (
                h5_total_local[0]
            )

        print()
        print(
            "H5 stored weight values:"
        )

        print(
            format_number(
                h5_parameter_count
            )
        )

        print()
        print(
            "H5 parameters in millions:"
        )

        print(
            format_millions(
                h5_parameter_count
            )
        )

except Exception as e:

    print()
    print(
        "H5 parameter analysis unavailable:"
    )

    print(e)


# ============================================================
# TEST DATA EVALUATION
# ============================================================

test_results = {}

if (
    os.path.exists(X_TEST_PATH)
    and
    os.path.exists(Y_TEST_PATH)
):

    print_section(
        "TEST DATA EVALUATION"
    )

    try:

        X_test = np.load(
            X_TEST_PATH,
            allow_pickle=False
        )

        y_test = np.load(
            Y_TEST_PATH,
            allow_pickle=False
        )

        print()
        print(
            "X_test shape:"
        )

        print(
            X_test.shape
        )

        print()
        print(
            "y_test shape:"
        )

        print(
            y_test.shape
        )

        # ----------------------------------------------------
        # Make sure channel exists
        # ----------------------------------------------------

        if X_test.ndim == 3:

            X_test = X_test[
                ..., np.newaxis
            ]

        X_test = X_test.astype(
            np.float32
        )

        # ----------------------------------------------------
        # TFLite single sample inference
        # ----------------------------------------------------

        predictions = []

        probabilities = []

        input_index = (
            input_details[0]["index"]
        )

        output_index = (
            output_details[0]["index"]
        )

        # Determine fixed/dynamic input.
        actual_input_shape = (
            interpreter
            .get_input_details()[0]["shape"]
        )

        for i in range(
            len(X_test)
        ):

            sample = X_test[
                i:i + 1
            ]

            try:

                interpreter.set_tensor(
                    input_index,
                    sample
                )

            except Exception:

                # Resize if necessary
                interpreter.resize_tensor_input(
                    input_index,
                    sample.shape,
                    strict=False
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

                input_index = (
                    input_details[0]["index"]
                )

                output_index = (
                    output_details[0]["index"]
                )

                interpreter.set_tensor(
                    input_index,
                    sample
                )

            interpreter.invoke()

            output = interpreter.get_tensor(
                output_index
            )

            probabilities.append(
                output[0]
            )

            predictions.append(
                int(
                    np.argmax(
                        output[0]
                    )
                )
            )

        y_pred = np.asarray(
            predictions,
            dtype=np.int64
        )

        probabilities = np.asarray(
            probabilities
        )

        # ----------------------------------------------------
        # Handle one-hot labels
        # ----------------------------------------------------

        if y_test.ndim > 1:

            y_true = np.argmax(
                y_test,
                axis=1
            )

        else:

            y_true = y_test.astype(
                np.int64
            )

        # ----------------------------------------------------
        # Metrics
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

        test_results = {

            "test_samples":
                int(len(y_true)),

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1)
        }

        print()
        print(
            f"Accuracy : {accuracy * 100:.2f}%"
        )

        print(
            f"Precision: {precision * 100:.2f}%"
        )

        print(
            f"Recall   : {recall * 100:.2f}%"
        )

        print(
            f"F1-score : {f1 * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )

        print()
        print(
            report
        )

        with open(
            os.path.join(
                REPORT_DIR,
                "classification_report.txt"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            f.write(report)

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        np.savetxt(
            os.path.join(
                REPORT_DIR,
                "confusion_matrix.csv"
            ),
            cm,
            fmt="%d",
            delimiter=","
        )

        if plt is not None:

            plt.figure(
                figsize=(10, 8)
            )

            if sns is not None:

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES
                )

            else:

                plt.imshow(
                    cm,
                    interpolation="nearest"
                )

                plt.colorbar()

                plt.xticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    rotation=45
                )

                plt.yticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES
                )

                for i in range(
                    cm.shape[0]
                ):

                    for j in range(
                        cm.shape[1]
                    ):

                        plt.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center"
                        )

            plt.xlabel(
                "Predicted Class"
            )

            plt.ylabel(
                "Actual Class"
            )

            plt.title(
                f"Baby Cry AI - TFLite Confusion Matrix\n"
                f"Accuracy = {accuracy * 100:.2f}%"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    REPORT_DIR,
                    "01_confusion_matrix.png"
                ),
                dpi=250
            )

            plt.close()

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        class_counts = []

        for i in range(
            num_classes
        ):

            class_counts.append(
                int(
                    np.sum(
                        y_true == i
                    )
                )
            )

        if plt is not None:

            plt.figure(
                figsize=(10, 6)
            )

            plt.bar(
                CLASS_NAMES,
                class_counts
            )

            plt.xlabel(
                "Class"
            )

            plt.ylabel(
                "Number of Test Samples"
            )

            plt.title(
                "Baby Cry AI - Test Dataset Distribution"
            )

            plt.xticks(
                rotation=30,
                ha="right"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    REPORT_DIR,
                    "02_test_class_distribution.png"
                ),
                dpi=250
            )

            plt.close()

        # ----------------------------------------------------
        # Per-class accuracy
        # ----------------------------------------------------

        per_class_accuracy = []

        for i in range(
            num_classes
        ):

            mask = (
                y_true == i
            )

            if np.sum(mask) > 0:

                class_acc = np.mean(
                    y_pred[mask] == i
                )

            else:

                class_acc = 0.0

            per_class_accuracy.append(
                float(class_acc)
            )

        if plt is not None:

            plt.figure(
                figsize=(10, 6)
            )

            plt.bar(
                CLASS_NAMES,
                [
                    x * 100
                    for x in per_class_accuracy
                ]
            )

            plt.xlabel(
                "Class"
            )

            plt.ylabel(
                "Accuracy (%)"
            )

            plt.title(
                "Baby Cry AI - Per Class Accuracy"
            )

            plt.xticks(
                rotation=30,
                ha="right"
            )

            plt.ylim(
                0,
                100
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    REPORT_DIR,
                    "03_per_class_accuracy.png"
                ),
                dpi=250
            )

            plt.close()

    except Exception as e:

        print()
        print(
            "TEST EVALUATION FAILED:"
        )

        print(e)


else:

    print_section(
        "TEST DATA"
    )

    print(
        "X_test.npy / y_test.npy were not found."
    )

    print(
        "Test accuracy cannot be calculated automatically."
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

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

        print_section(
            "TRAINING HISTORY"
        )

        keys = list(
            history_data.keys()
        )

        print(
            "Available metrics:"
        )

        for key in keys:

            print(
                " ",
                key
            )

        if plt is not None:

            # ------------------------------------------------
            # Accuracy graph
            # ------------------------------------------------

            if (
                "accuracy" in history_data
                and
                "val_accuracy"
                in history_data
            ):

                plt.figure(
                    figsize=(10, 6)
                )

                plt.plot(
                    history_data["accuracy"],
                    label="Training Accuracy"
                )

                plt.plot(
                    history_data["val_accuracy"],
                    label="Validation Accuracy"
                )

                plt.xlabel(
                    "Epoch"
                )

                plt.ylabel(
                    "Accuracy"
                )

                plt.title(
                    "Baby Cry AI - Training vs Validation Accuracy"
                )

                plt.legend()

                plt.grid(
                    True,
                    alpha=0.3
                )

                plt.tight_layout()

                plt.savefig(
                    os.path.join(
                        REPORT_DIR,
                        "04_training_accuracy.png"
                    ),
                    dpi=250
                )

                plt.close()

            # ------------------------------------------------
            # Loss graph
            # ------------------------------------------------

            if (
                "loss" in history_data
                and
                "val_loss"
                in history_data
            ):

                plt.figure(
                    figsize=(10, 6)
                )

                plt.plot(
                    history_data["loss"],
                    label="Training Loss"
                )

                plt.plot(
                    history_data["val_loss"],
                    label="Validation Loss"
                )

                plt.xlabel(
                    "Epoch"
                )

                plt.ylabel(
                    "Loss"
                )

                plt.title(
                    "Baby Cry AI - Training vs Validation Loss"
                )

                plt.legend()

                plt.grid(
                    True,
                    alpha=0.3
                )

                plt.tight_layout()

                plt.savefig(
                    os.path.join(
                        REPORT_DIR,
                        "05_training_loss.png"
                    ),
                    dpi=250
                )

                plt.close()

    except Exception as e:

        print(
            "Could not read training history:",
            e
        )


# ============================================================
# NORMALIZATION INFORMATION
# ============================================================

normalization_info = None

if os.path.exists(
    NORMALIZATION_PATH
):

    try:

        with open(
            NORMALIZATION_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            normalization_info = json.load(f)

    except Exception:
        pass


# ============================================================
# COMPLETE MODEL STATISTICS
# ============================================================

model_statistics = {

    "model_name":
        "BabyAudioStrongCNN",

    "tflite_model":
        os.path.basename(
            TFLITE_MODEL_PATH
        ),

    "h5_model":
        os.path.basename(
            H5_MODEL_PATH
        )
        if os.path.exists(
            H5_MODEL_PATH
        )
        else None,

    "tflite_size_bytes":
        int(tflite_size),

    "tflite_size_MB":
        round(
            tflite_size / 1024 / 1024,
            4
        ),

    "h5_size_MB":
        round(
            h5_size / 1024 / 1024,
            4
        )
        if h5_size is not None
        else None,

    "tensorflow_version":
        tf.__version__,

    "input_shape":
        input_shape,

    "output_shape":
        output_shape,

    "number_of_classes":
        int(num_classes),

    "classes":
        CLASS_NAMES,

    "estimated_tflite_parameters":
        int(total_parameters),

    "estimated_tflite_parameters_M":
        round(
            total_parameters / 1_000_000,
            6
        ),

    "h5_stored_weight_values":
        int(h5_parameter_count)
        if h5_parameter_count is not None
        else None,

    "h5_stored_weight_values_M":
        round(
            h5_parameter_count / 1_000_000,
            6
        )
        if h5_parameter_count is not None
        else None,

    "tflite_nodes":
        int(node_count),

    "activation_tensors":
        int(
            len(
                activation_tensors
            )
        ),

    "total_activation_neurons":
        int(
            total_activation_neurons
        ),

    "total_activation_neurons_M":
        round(
            total_activation_neurons / 1_000_000,
            6
        ),

    "maximum_activation_neurons":
        int(
            max_activation_neurons
        ),

    "operator_distribution":
        operator_counts,

    "test_results":
        test_results,

    "normalization":
        normalization_info
}


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    REPORT_DIR,
    "complete_model_report.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_statistics,
        f,
        indent=4
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

txt_path = os.path.join(
    REPORT_DIR,
    "complete_model_report.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BABY CRY AI COMPLETE MODEL REPORT\n"
    )

    f.write(
        "=" * 90
        + "\n\n"
    )

    f.write(
        f"Model                 : BabyAudioStrongCNN\n"
    )

    f.write(
        f"TFLite size           : {format_mb(tflite_size)}\n"
    )

    if h5_size is not None:

        f.write(
            f"H5 size               : {format_mb(h5_size)}\n"
        )

    f.write(
        f"TensorFlow             : {tf.__version__}\n"
    )

    f.write(
        f"Input shape            : {input_shape}\n"
    )

    f.write(
        f"Output shape           : {output_shape}\n"
    )

    f.write(
        f"Classes                : {num_classes}\n"
    )

    f.write(
        f"TFLite parameters      : {total_parameters:,}\n"
    )

    f.write(
        f"TFLite parameters (M)  : {total_parameters / 1_000_000:.4f} M\n"
    )

    if h5_parameter_count is not None:

        f.write(
            f"H5 stored weights     : {h5_parameter_count:,}\n"
        )

        f.write(
            f"H5 weights (M)        : "
            f"{h5_parameter_count / 1_000_000:.4f} M\n"
        )

    f.write(
        f"TFLite nodes/operators: {node_count:,}\n"
    )

    f.write(
        f"Activation tensors     : "
        f"{len(activation_tensors):,}\n"
    )

    f.write(
        f"Activation neurons     : "
        f"{total_activation_neurons:,}\n"
    )

    f.write(
        f"Activation neurons (M) : "
        f"{total_activation_neurons / 1_000_000:.4f} M\n"
    )

    f.write(
        f"Maximum activation     : "
        f"{max_activation_neurons:,}\n"
    )

    f.write(
        "\nCLASS MAPPING\n"
    )

    f.write(
        "-" * 50
        + "\n"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        f.write(
            f"{i} -> {name}\n"
        )

    f.write(
        "\nOPERATOR DISTRIBUTION\n"
    )

    f.write(
        "-" * 50
        + "\n"
    )

    for name, count in sorted(
        operator_counts.items()
    ):

        f.write(
            f"{name:<35} {count}\n"
        )

    if test_results:

        f.write(
            "\nTEST PERFORMANCE\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"Test samples : "
            f"{test_results['test_samples']}\n"
        )

        f.write(
            f"Accuracy     : "
            f"{test_results['accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"Precision    : "
            f"{test_results['precision'] * 100:.2f}%\n"
        )

        f.write(
            f"Recall       : "
            f"{test_results['recall'] * 100:.2f}%\n"
        )

        f.write(
            f"F1-score     : "
            f"{test_results['f1_score'] * 100:.2f}%\n"
        )


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print_section(
    "FINAL BABY CRY AI MODEL SPECIFICATION"
)

print()
print(
    "MODEL"
)

print(
    "----------------------------------------"
)

print(
    f"Model name             : BabyAudioStrongCNN"
)

print(
    f"TFLite file size       : {format_mb(tflite_size)}"
)

if h5_size is not None:

    print(
        f"H5 file size           : {format_mb(h5_size)}"
    )

print()
print(
    "ARCHITECTURE"
)

print(
    "----------------------------------------"
)

print(
    f"Input shape             : {input_shape}"
)

print(
    f"Output shape            : {output_shape}"
)

print(
    f"Classes                 : {num_classes}"
)

print(
    f"TFLite nodes/operators : {node_count:,}"
)

print()
print(
    "PARAMETERS"
)

print(
    "----------------------------------------"
)

print(
    f"Parameters              : {total_parameters:,}"
)

print(
    f"Parameters              : {total_parameters / 1_000_000:.4f} Million"
)

if h5_parameter_count is not None:

    print(
        f"H5 stored weights      : "
        f"{h5_parameter_count:,}"
    )

    print(
        f"H5 stored weights      : "
        f"{h5_parameter_count / 1_000_000:.4f} Million"
    )

print()
print(
    "NEURONS / ACTIVATIONS"
)

print(
    "----------------------------------------"
)

print(
    f"Activation tensors      : "
    f"{len(activation_tensors):,}"
)

print(
    f"Activation neurons      : "
    f"{total_activation_neurons:,}"
)

print(
    f"Activation neurons      : "
    f"{total_activation_neurons / 1_000_000:.4f} Million"
)

print(
    f"Maximum activation      : "
    f"{max_activation_neurons:,}"
)

if test_results:

    print()
    print(
        "TEST PERFORMANCE"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Test samples            : "
        f"{test_results['test_samples']:,}"
    )

    print(
        f"Accuracy                : "
        f"{test_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision               : "
        f"{test_results['precision'] * 100:.2f}%"
    )

    print(
        f"Recall                  : "
        f"{test_results['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score                : "
        f"{test_results['f1_score'] * 100:.2f}%"
    )


# ============================================================
# SAVE A SIMPLE MODEL STATISTICS IMAGE
# ============================================================

if plt is not None:

    image_path = os.path.join(
        REPORT_DIR,
        "00_model_statistics.png"
    )

    fig = plt.figure(
        figsize=(12, 8)
    )

    ax = fig.add_subplot(
        111
    )

    ax.axis(
        "off"
    )

    lines = [

        "BABY CRY AI MODEL - COMPLETE STATISTICS",

        "",

        f"Model: BabyAudioStrongCNN",

        f"TFLite Size: {format_mb(tflite_size)}",

        f"Input: {input_shape}",

        f"Output: {output_shape}",

        f"Classes: {num_classes}",

        "",

        f"PARAMETERS: {total_parameters:,}",

        f"PARAMETERS: {total_parameters / 1_000_000:.4f} MILLION",

        f"TFLITE NODES / OPERATORS: {node_count:,}",

        "",

        f"ACTIVATION NEURONS: {total_activation_neurons:,}",

        f"ACTIVATION NEURONS: "
        f"{total_activation_neurons / 1_000_000:.4f} MILLION",

        f"MAX ACTIVATION: {max_activation_neurons:,}",

    ]

    if h5_parameter_count is not None:

        lines.extend(
            [
                "",
                f"H5 STORED WEIGHTS: "
                f"{h5_parameter_count:,}",

                f"H5 STORED WEIGHTS: "
                f"{h5_parameter_count / 1_000_000:.4f} MILLION"
            ]
        )

    if test_results:

        lines.extend(
            [
                "",
                f"TEST ACCURACY: "
                f"{test_results['accuracy'] * 100:.2f}%",

                f"PRECISION: "
                f"{test_results['precision'] * 100:.2f}%",

                f"RECALL: "
                f"{test_results['recall'] * 100:.2f}%",

                f"F1 SCORE: "
                f"{test_results['f1_score'] * 100:.2f}%"
            ]
        )

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=15,
        family="monospace"
    )

    plt.tight_layout()

    plt.savefig(
        image_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FINISHED
# ============================================================

print()
print(separator())
print("REPORT GENERATION COMPLETE")
print(separator())

print()
print(
    "All report files saved to:"
)

print(
    REPORT_DIR
)

print()
print(
    "Important image files:"
)

if plt is not None:

    print(
        "  00_model_statistics.png"
    )

    print(
        "  01_confusion_matrix.png"
    )

    print(
        "  02_test_class_distribution.png"
    )

    print(
        "  03_per_class_accuracy.png"
    )

    print(
        "  04_training_accuracy.png"
    )

    print(
        "  05_training_loss.png"
    )

print()
print(
    "Text:"
)

print(
    "  complete_model_report.txt"
)

print()
print(
    "JSON:"
)

print(
    "  complete_model_report.json"
)

print()
print(separator())