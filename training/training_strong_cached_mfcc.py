# ============================================================
# BABY AUDIO CLASSIFICATION
# STRONG CNN + python_speech_features MFCC + MFCC CACHE
#
# Designed for:
#   Windows training PC
#   Python 3.9.2
#   TensorFlow 2.10.x
#   GTX 1660 Super 6GB
#
# Deployment:
#   Raspberry Pi Zero 2 W
#   Python 3.9.2
#   tflite_runtime
#
# IMPORTANT:
# The MFCC extraction is intentionally matched to the old
# Raspberry Pi inference code.
#
# MFCC:
#   samplerate = original audio rate
#   numcep     = 24
#   winlen     = 0.025
#   winstep    = 0.01
#   nfft       = 1024
#
# Final shape:
#   (500, 24, 1)
#
# ============================================================


# ============================================================
# 0. ENVIRONMENT SETTINGS
# ============================================================

import os

# IMPORTANT:
# Do NOT enable TensorFlow deterministic GPU operations.
# Your previous error:
#
# "Deterministic GPU implementation of unsorted segment
# reduction op not available"
#
# was caused by deterministic GPU execution.
os.environ["TF_DETERMINISTIC_OPS"] = "0"

# Reduce TensorFlow console noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# ============================================================
# 1. IMPORTS
# ============================================================

import sys
import json
import time
import hashlib
import random
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from scipy.signal import resample

from python_speech_features import mfcc as psf_mfcc

import tensorflow as tf

from tensorflow.keras import (
    layers,
    models,
    regularizers,
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATASET_ROOT = Path("combined")

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = Path("models")

CACHE_DIR = OUTPUT_DIR / "feature_cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


BEST_MODEL_PATH = OUTPUT_DIR / "baby_audio_strong_best.h5"

FINAL_MODEL_PATH = OUTPUT_DIR / "baby_audio_strong_final.h5"

TFLITE_MODEL_PATH = OUTPUT_DIR / "baby_audio_strong.tflite"

LABEL_MAP_PATH = OUTPUT_DIR / "label_map.json"

NORMALIZATION_PATH = OUTPUT_DIR / "normalization.json"

HISTORY_PATH = OUTPUT_DIR / "training_history.json"

REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

INFO_PATH = OUTPUT_DIR / "deployment_info.json"


# ------------------------------------------------------------
# Audio / MFCC
# ------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000

NUM_MFCC = 24

WINLEN = 0.025

WINSTEP = 0.01

NFFT = 1024

MAX_FRAMES = 500


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 8

EPOCHS = 100

INITIAL_LEARNING_RATE = 0.0003

RANDOM_SEED = 42


# ------------------------------------------------------------
# Cache version
# ------------------------------------------------------------

FEATURE_VERSION = (
    "PSF_MFCC_V1"
    "_SR16000"
    "_NUMCEP24"
    "_WINLEN0.025"
    "_WINSTEP0.01"
    "_NFFT1024"
    "_FRAMES500"
)


# ============================================================
# 3. RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 4. GPU CONFIGURATION
# ============================================================

print("=" * 70)
print("TensorFlow:", tf.__version__)
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected:")

    for gpu in gpus:

        print("  ", gpu)

        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
        except Exception as e:
            print(
                "WARNING: Could not enable memory growth:",
                e
            )

else:

    print("WARNING: No GPU detected.")
    print("Training will use CPU.")


# ============================================================
# 5. CPU THREAD SETTINGS
# ============================================================

try:

    tf.config.threading.set_inter_op_parallelism_threads(2)

    tf.config.threading.set_intra_op_parallelism_threads(4)

except Exception:
    pass


# ============================================================
# 6. CHECK DATASET
# ============================================================

print("\nChecking dataset...")

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR.resolve()}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found:\n{VAL_DIR.resolve()}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test folder not found:\n{TEST_DIR.resolve()}"
    )


# ============================================================
# 7. AUDIO EXTENSIONS
# ============================================================

AUDIO_EXTENSIONS = {
    ".wav",
    ".wave",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
}


# ============================================================
# 8. FIND CLASS NAMES
# ============================================================

def get_class_names():

    classes = []

    for item in TRAIN_DIR.iterdir():

        if item.is_dir():

            classes.append(item.name)

    classes = sorted(classes)

    if len(classes) < 2:

        raise RuntimeError(
            "Less than 2 class folders were found."
        )

    return classes


CLASS_NAMES = get_class_names()

NUM_CLASSES = len(CLASS_NAMES)


print("\nClasses:")

for i, name in enumerate(CLASS_NAMES):

    print(f"  {i}: {name}")

print("\nNumber of classes:", NUM_CLASSES)


# ============================================================
# 9. VERIFY VALIDATION / TEST CLASSES
# ============================================================

for split_dir in [VAL_DIR, TEST_DIR]:

    existing = sorted(
        [
            p.name
            for p in split_dir.iterdir()
            if p.is_dir()
        ]
    )

    if existing != CLASS_NAMES:

        raise RuntimeError(
            f"\nClass mismatch in {split_dir}\n"
            f"Expected: {CLASS_NAMES}\n"
            f"Found:    {existing}"
        )


# ============================================================
# 10. SAVE LABEL MAP
# ============================================================

label_map = {
    str(i): name
    for i, name in enumerate(CLASS_NAMES)
}

with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:

    json.dump(
        label_map,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 11. FIND AUDIO FILES
# ============================================================

def find_audio_files(split_dir):

    files = []
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = split_dir / class_name

        for path in class_dir.rglob("*"):

            if (
                path.is_file()
                and path.suffix.lower() in AUDIO_EXTENSIONS
            ):

                files.append(path)
                labels.append(class_index)

    return files, np.asarray(labels, dtype=np.int64)


# ============================================================
# 12. AUDIO LOADING
# ============================================================

def load_audio_exact(path):

    """
    Load audio.

    Important:
    No amplitude normalization is performed.

    This intentionally keeps the preprocessing close to the
    old Raspberry Pi pipeline.
    """

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # Stereo -> mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    # Remove NaN / Inf
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    if len(audio) == 0:

        raise ValueError(
            f"Empty audio file: {path}"
        )

    # --------------------------------------------------------
    # Resample exactly when required.
    #
    # Your Pi currently records 48 kHz and resamples to
    # 16 kHz before MFCC extraction.
    #
    # scipy.signal.resample is used here as well.
    # --------------------------------------------------------

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
                f"Invalid target length for {path}"
            )

        audio = resample(
            audio,
            target_length
        ).astype(np.float32)

    return audio


# ============================================================
# 13. MFCC EXTRACTION
# ============================================================

def extract_mfcc_exact(path):

    """
    EXACT MFCC settings used by your old Raspberry Pi code.

    Old Pi code:

        mfcc(
            signal,
            samplerate=sr,
            numcep=24,
            winlen=0.025,
            winstep=0.01,
            nfft=1024
        )

    """

    audio = load_audio_exact(path)

    features = psf_mfcc(
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

    # --------------------------------------------------------
    # Force exactly 500 frames
    # --------------------------------------------------------

    if features.shape[0] < MAX_FRAMES:

        pad_amount = (
            MAX_FRAMES
            - features.shape[0]
        )

        features = np.pad(
            features,
            (
                (0, pad_amount),
                (0, 0)
            ),
            mode="constant"
        )

    else:

        features = features[:MAX_FRAMES, :]

    # Safety
    if features.shape != (
        MAX_FRAMES,
        NUM_MFCC
    ):

        raise RuntimeError(
            f"Unexpected MFCC shape {features.shape}"
        )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features.astype(np.float32)


# ============================================================
# 14. CACHE KEY
# ============================================================

def get_cache_key(audio_path):

    """
    Cache is invalidated automatically when:
      - file changes
      - file size changes
      - MFCC configuration changes
    """

    stat = audio_path.stat()

    text = (
        str(audio_path.resolve())
        + "|"
        + str(stat.st_size)
        + "|"
        + str(stat.st_mtime_ns)
        + "|"
        + FEATURE_VERSION
    )

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 15. CACHE PATH
# ============================================================

def get_cache_path(split_name, audio_path):

    split_cache = CACHE_DIR / split_name

    split_cache.mkdir(
        parents=True,
        exist_ok=True
    )

    key = get_cache_key(audio_path)

    return split_cache / (
        key + ".npy"
    )


# ============================================================
# 16. GET MFCC WITH CACHE
# ============================================================

def get_mfcc_cached(
    split_name,
    audio_path
):

    cache_path = get_cache_path(
        split_name,
        audio_path
    )

    # --------------------------------------------------------
    # Cache hit
    # --------------------------------------------------------

    if cache_path.exists():

        try:

            features = np.load(
                str(cache_path),
                allow_pickle=False
            )

            if features.shape == (
                MAX_FRAMES,
                NUM_MFCC
            ):

                return features.astype(
                    np.float32,
                    copy=False
                ), True

        except Exception:

            # Corrupt cache -> regenerate
            try:
                cache_path.unlink()
            except Exception:
                pass

    # --------------------------------------------------------
    # Cache miss
    # --------------------------------------------------------

    features = extract_mfcc_exact(
        audio_path
    )

    # Atomic save
    temporary = cache_path.with_suffix(
        ".tmp.npy"
    )

    np.save(
        str(temporary),
        features,
        allow_pickle=False
    )

    os.replace(
        str(temporary),
        str(cache_path)
    )

    return features, False


# ============================================================
# 17. LOAD COMPLETE DATASET
# ============================================================

def load_dataset(
    split_name,
    split_dir
):

    print("\n" + "=" * 70)
    print(f"Loading {split_name.upper()} dataset")
    print("=" * 70)

    files, labels = find_audio_files(
        split_dir
    )

    if len(files) == 0:

        raise RuntimeError(
            f"No audio files found in {split_dir}"
        )

    print(
        f"Audio files found: {len(files)}"
    )

    X = np.empty(
        (
            len(files),
            MAX_FRAMES,
            NUM_MFCC
        ),
        dtype=np.float32
    )

    cache_hits = 0
    cache_misses = 0

    start_time = time.time()

    for i, path in enumerate(files):

        try:

            features, cached = get_mfcc_cached(
                split_name,
                path
            )

            X[i] = features

            if cached:
                cache_hits += 1
            else:
                cache_misses += 1

        except Exception as e:

            print(
                f"\nERROR processing:\n{path}\n{e}"
            )

            raise

        if (
            (i + 1) % 50 == 0
            or i == len(files) - 1
        ):

            elapsed = time.time() - start_time

            rate = (
                (i + 1) / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(files) - (i + 1)
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"\rProcessed "
                f"{i+1}/{len(files)} | "
                f"Cache hit: {cache_hits} | "
                f"New: {cache_misses} | "
                f"ETA: {eta:.1f}s",
                end=""
            )

    print()

    print(
        f"Cache hits   : {cache_hits}"
    )

    print(
        f"Cache misses : {cache_misses}"
    )

    print(
        f"X shape      : {X.shape}"
    )

    print(
        f"Memory       : "
        f"{X.nbytes / 1024 / 1024:.1f} MB"
    )

    return X, labels, files


# ============================================================
# 18. LOAD TRAIN / VALIDATION / TEST
# ============================================================

X_train, y_train, train_files = load_dataset(
    "train",
    TRAIN_DIR
)

X_val, y_val, val_files = load_dataset(
    "validation",
    VAL_DIR
)

X_test, y_test, test_files = load_dataset(
    "test",
    TEST_DIR
)


# ============================================================
# 19. PRINT CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    labels
):

    print("\n" + name)

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(labels == index)
        )

        print(
            f"  {class_name:<20} "
            f"{count}"
        )


print_distribution(
    "TRAIN DISTRIBUTION",
    y_train
)

print_distribution(
    "VALIDATION DISTRIBUTION",
    y_val
)

print_distribution(
    "TEST DISTRIBUTION",
    y_test
)


# ============================================================
# 20. ADD CHANNEL DIMENSION
# ============================================================

X_train = X_train[..., np.newaxis]

X_val = X_val[..., np.newaxis]

X_test = X_test[..., np.newaxis]


print("\nFinal input shape:")

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# 21. CALCULATE NORMALIZATION
# ============================================================

print("\nCalculating MFCC normalization...")

# Calculate per-MFCC mean/std.
#
# Shape:
#   (1, 1, 24, 1)
#
# Normalization is later embedded directly into the model.

mfcc_mean = np.mean(
    X_train,
    axis=(0, 1),
    keepdims=True
).astype(np.float32)

mfcc_std = np.std(
    X_train,
    axis=(0, 1),
    keepdims=True
).astype(np.float32)

mfcc_std = np.maximum(
    mfcc_std,
    1e-6
)


print(
    "Mean shape:",
    mfcc_mean.shape
)

print(
    "Std shape:",
    mfcc_std.shape
)


# ============================================================
# 22. SAVE NORMALIZATION
# ============================================================

with open(
    NORMALIZATION_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "mean": mfcc_mean.reshape(-1).tolist(),
            "std": mfcc_std.reshape(-1).tolist()
        },
        f,
        indent=4
    )


# ============================================================
# 23. FIXED STANDARDIZATION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable()
class FixedStandardization(
    layers.Layer
):

    """
    Normalization layer whose values are fixed.

    IMPORTANT:
    Mean/std are stored as ordinary NumPy/Python values
    in get_config().

    This avoids the EagerTensor -> JSON serialization problem
    that happened with the previous Normalization implementation.
    """

    def __init__(
        self,
        mean,
        std,
        **kwargs
    ):

        super().__init__(**kwargs)

        mean = np.asarray(
            mean,
            dtype=np.float32
        )

        std = np.asarray(
            std,
            dtype=np.float32
        )

        self.mean_values = (
            mean.reshape(
                1,
                1,
                NUM_MFCC,
                1
            )
        )

        self.std_values = (
            np.maximum(
                std.reshape(
                    1,
                    1,
                    NUM_MFCC,
                    1
                ),
                1e-6
            )
        )

    def build(self, input_shape):

        self.mean_weight = self.add_weight(
            name="mean",
            shape=self.mean_values.shape,
            initializer=tf.keras.initializers.Constant(
                self.mean_values
            ),
            trainable=False
        )

        self.std_weight = self.add_weight(
            name="std",
            shape=self.std_values.shape,
            initializer=tf.keras.initializers.Constant(
                self.std_values
            ),
            trainable=False
        )

        super().build(input_shape)

    def call(self, inputs):

        inputs = tf.cast(
            inputs,
            tf.float32
        )

        return (
            inputs - self.mean_weight
        ) / self.std_weight

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "mean": self.mean_values.reshape(
                    -1
                ).tolist(),

                "std": self.std_values.reshape(
                    -1
                ).tolist()
            }
        )

        return config


# ============================================================
# 24. MFCC AUGMENTATION
# ============================================================

def augment_mfcc(
    x,
    y
):

    """
    Feature-domain augmentation.

    This is ONLY applied to training data.

    Validation and test remain untouched.
    """

    x = tf.cast(
        x,
        tf.float32
    )

    # --------------------------------------------------------
    # Small random MFCC noise
    # --------------------------------------------------------

    random_value = tf.random.uniform(())

    def add_noise():

        std = tf.math.reduce_std(x)

        noise = tf.random.normal(
            tf.shape(x),
            mean=0.0,
            stddev=tf.maximum(
                std * 0.015,
                0.001
            )
        )

        return x + noise

    x = tf.cond(
        random_value < 0.35,
        add_noise,
        lambda: x
    )

    # --------------------------------------------------------
    # Time masking
    # --------------------------------------------------------

    random_value = tf.random.uniform(())

    def time_mask():

        width = tf.random.uniform(
            [],
            minval=5,
            maxval=31,
            dtype=tf.int32
        )

        start = tf.random.uniform(
            [],
            minval=0,
            maxval=MAX_FRAMES - 30,
            dtype=tf.int32
        )

        indices = tf.range(
            MAX_FRAMES
        )

        mask_1d = (
            (indices >= start)
            &
            (indices < start + width)
        )

        mask = tf.reshape(
            mask_1d,
            [MAX_FRAMES, 1, 1]
        )

        mean_frame = tf.reduce_mean(
            x,
            axis=0,
            keepdims=True
        )

        return tf.where(
            mask,
            mean_frame,
            x
        )

    x = tf.cond(
        random_value < 0.20,
        time_mask,
        lambda: x
    )

    # --------------------------------------------------------
    # Frequency masking
    # --------------------------------------------------------

    random_value = tf.random.uniform(())

    def frequency_mask():

        width = tf.random.uniform(
            [],
            minval=1,
            maxval=5,
            dtype=tf.int32
        )

        start = tf.random.uniform(
            [],
            minval=0,
            maxval=NUM_MFCC - 5,
            dtype=tf.int32
        )

        indices = tf.range(
            NUM_MFCC
        )

        mask_1d = (
            (indices >= start)
            &
            (indices < start + width)
        )

        mask = tf.reshape(
            mask_1d,
            [1, NUM_MFCC, 1]
        )

        mean_value = tf.reduce_mean(
            x,
            axis=1,
            keepdims=True
        )

        return tf.where(
            mask,
            mean_value,
            x
        )

    x = tf.cond(
        random_value < 0.15,
        frequency_mask,
        lambda: x
    )

    return x, y


# ============================================================
# 25. CREATE TF.DATA DATASETS
# ============================================================

print("\nCreating tf.data datasets...")


train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train
    )
)

train_dataset = train_dataset.shuffle(
    buffer_size=min(
        len(X_train),
        5000
    ),
    seed=RANDOM_SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    augment_mfcc,
    num_parallel_calls=2
)

train_dataset = train_dataset.batch(
    BATCH_SIZE,
    drop_remainder=False
)

train_dataset = train_dataset.prefetch(
    1
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_val,
        y_val
    )
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    1
)


# ============================================================
# 26. CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train
)

class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"  {class_name:<20} "
        f"{class_weights[i]:.4f}"
    )


# ============================================================
# 27. BUILD STRONG CNN
# ============================================================

def residual_separable_block(
    x,
    filters,
    dropout_rate
):

    """
    Lightweight residual separable convolution block.

    Good balance between:
      - accuracy
      - memory
      - TFLite compatibility
      - Raspberry Pi speed
    """

    input_channels = x.shape[-1]

    # --------------------------------------------------------
    # Shortcut
    # --------------------------------------------------------

    shortcut = x

    if (
        input_channels is None
        or int(input_channels) != filters
    ):

        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(
            shortcut
        )

    # --------------------------------------------------------
    # Main path
    # --------------------------------------------------------

    y = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    y = layers.BatchNormalization()(y)

    y = layers.ReLU()(y)

    y = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(y)

    y = layers.BatchNormalization()(y)

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    y = layers.Add()(
        [
            y,
            shortcut
        ]
    )

    y = layers.ReLU()(y)

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    y = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(y)

    y = layers.Dropout(
        dropout_rate
    )(y)

    return y


# ============================================================
# 28. MODEL
# ============================================================

input_shape = (
    MAX_FRAMES,
    NUM_MFCC,
    1
)


inputs = layers.Input(
    shape=input_shape,
    name="mfcc_input"
)


# ------------------------------------------------------------
# Embedded normalization
# ------------------------------------------------------------

x = FixedStandardization(
    mean=mfcc_mean.reshape(-1),
    std=mfcc_std.reshape(-1),
    name="mfcc_standardization"
)(inputs)


# ------------------------------------------------------------
# Initial convolution
# ------------------------------------------------------------

x = layers.Conv2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)

x = layers.ReLU()(x)


# ------------------------------------------------------------
# Residual blocks
# ------------------------------------------------------------

x = residual_separable_block(
    x,
    filters=32,
    dropout_rate=0.08
)


x = residual_separable_block(
    x,
    filters=64,
    dropout_rate=0.10
)


x = residual_separable_block(
    x,
    filters=96,
    dropout_rate=0.12
)


x = residual_separable_block(
    x,
    filters=128,
    dropout_rate=0.15
)


# ------------------------------------------------------------
# Global pooling
# ------------------------------------------------------------

avg_pool = layers.GlobalAveragePooling2D()(x)

max_pool = layers.GlobalMaxPooling2D()(x)

x = layers.Concatenate()(
    [
        avg_pool,
        max_pool
    ]
)


# ------------------------------------------------------------
# Dense classifier
# ------------------------------------------------------------

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
    activation="softmax",
    name="prediction"
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="BabyAudioStrongCNN"
)


# ============================================================
# 29. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

model.summary()


# ============================================================
# 30. COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=INITIAL_LEARNING_RATE,
    clipnorm=1.0
)


model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 31. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    mode="max",
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(BEST_MODEL_PATH),

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    save_weights_only=False,

    verbose=1
)


csv_logger = tf.keras.callbacks.CSVLogger(
    str(
        OUTPUT_DIR
        / "training_log.csv"
    ),
    append=False
)


callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr,
    csv_logger
]


# ============================================================
# 32. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "MFCC:",
    "python_speech_features"
)

print(
    "Input:",
    input_shape
)

print(
    "GPU:",
    gpus if gpus else "CPU"
)

print()


training_start = time.time()


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


training_time = (
    time.time()
    - training_start
)


print(
    "\nTraining completed in "
    f"{training_time / 60:.2f} minutes."
)


# ============================================================
# 33. SAVE HISTORY SAFELY
# ============================================================

history_dict = {}

for key, values in history.history.items():

    history_dict[key] = [
        float(v)
        for v in values
    ]


with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_dict,
        f,
        indent=4
    )


# ============================================================
# 34. LOAD BEST MODEL
# ============================================================

print("\nLoading best checkpoint...")

best_model = tf.keras.models.load_model(
    str(BEST_MODEL_PATH),
    custom_objects={
        "FixedStandardization":
            FixedStandardization
    },
    compile=False
)


# ============================================================
# 35. COMPILE BEST MODEL
# ============================================================

best_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 36. SAVE FINAL H5
# ============================================================

best_model.save(
    str(FINAL_MODEL_PATH),
    include_optimizer=False
)


print(
    "\nFinal H5 model saved:"
)

print(
    FINAL_MODEL_PATH.resolve()
)


# ============================================================
# 37. VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

val_loss, val_accuracy = best_model.evaluate(
    X_val,
    y_val,
    batch_size=BATCH_SIZE,
    verbose=1
)


print(
    f"\nValidation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# 38. TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


print(
    f"\nTest accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 39. TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

test_probabilities = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


test_predictions = np.argmax(
    test_probabilities,
    axis=1
)


# ============================================================
# 40. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    test_predictions,
    target_names=CLASS_NAMES,
    digits=4
)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# 41. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Blues"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    f"Baby Audio Test Confusion Matrix\n"
    f"Accuracy = {test_accuracy * 100:.2f}%"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200
)

plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    CONFUSION_MATRIX_PATH.resolve()
)


# ============================================================
# 42. TFLITE CONVERSION
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING TO TFLITE")
print("=" * 70)


# Make sure TensorFlow graph is clean
tf.keras.backend.clear_session()


# Reload once more
best_model = tf.keras.models.load_model(
    str(BEST_MODEL_PATH),
    custom_objects={
        "FixedStandardization":
            FixedStandardization
    },
    compile=False
)


converter = tf.lite.TFLiteConverter.from_keras_model(
    best_model
)


# Dynamic-range optimization.
#
# Important:
# Input remains FLOAT32.
#
# This is safer for your existing Pi MFCC pipeline.
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]


tflite_model = converter.convert()


with open(
    TFLITE_MODEL_PATH,
    "wb"
) as f:

    f.write(tflite_model)


print(
    "\nTFLite model saved:"
)

print(
    TFLITE_MODEL_PATH.resolve()
)

print(
    "TFLite size:",
    f"{len(tflite_model) / 1024 / 1024:.2f} MB"
)


# ============================================================
# 43. TFLITE TEST
# ============================================================

print("\n" + "=" * 70)
print("TESTING TFLITE MODEL")
print("=" * 70)


def evaluate_tflite(
    model_path,
    X,
    y
):

    """
    Fast TFLite evaluation.

    Attempts batch inference first.

    If the TFLite model does not allow dynamic batch resizing,
    it automatically falls back to one-sample inference.
    """

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=4
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_index = (
        input_details[0]["index"]
    )

    output_index = (
        output_details[0]["index"]
    )

    input_shape_now = (
        input_details[0]["shape"]
    )

    input_dtype = (
        input_details[0]["dtype"]
    )

    print(
        "\nTFLite input shape:",
        input_shape_now
    )

    print(
        "TFLite input dtype:",
        input_dtype
    )

    print(
        "TFLite output shape:",
        output_details[0]["shape"]
    )

    print(
        "TFLite output dtype:",
        output_details[0]["dtype"]
    )

    # --------------------------------------------------------
    # Make sure input is float32
    # --------------------------------------------------------

    X = X.astype(
        np.float32,
        copy=False
    )

    predictions = []

    # --------------------------------------------------------
    # Try dynamic batch
    # --------------------------------------------------------

    dynamic_batch_supported = False

    try:

        shape_signature = (
            input_details[0]
            .get("shape_signature", None)
        )

        print(
            "Shape signature:",
            shape_signature
        )

        if (
            shape_signature is not None
            and shape_signature[0] == -1
        ):

            dynamic_batch_supported = True

    except Exception:

        pass


    # --------------------------------------------------------
    # Fast batch evaluation
    # --------------------------------------------------------

    if dynamic_batch_supported:

        batch_size = 64

        total = len(X)

        start_time = time.time()

        for start in range(
            0,
            total,
            batch_size
        ):

            end = min(
                start + batch_size,
                total
            )

            batch = X[start:end]

            try:

                interpreter.resize_tensor_input(
                    input_index,
                    [
                        len(batch),
                        MAX_FRAMES,
                        NUM_MFCC,
                        1
                    ],
                    strict=False
                )

                interpreter.allocate_tensors()

                input_details = (
                    interpreter.get_input_details()
                )

                output_details = (
                    interpreter.get_output_details()
                )

                input_index = (
                    input_details[0]["index"]
                )

                output_index = (
                    output_details[0]["index"]
                )

                interpreter.set_tensor(
                    input_index,
                    batch
                )

                interpreter.invoke()

                output = interpreter.get_tensor(
                    output_index
                )

                predictions.extend(
                    np.argmax(
                        output,
                        axis=1
                    ).tolist()
                )

            except Exception as e:

                print(
                    "\nBatch TFLite inference failed:"
                )

                print(e)

                print(
                    "Falling back to single-sample mode..."
                )

                predictions = []

                dynamic_batch_supported = False

                break


            done = end

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                done / elapsed
                if elapsed > 0
                else 0
            )

            remaining = total - done

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"\rTFLite testing "
                f"{done}/{total} | "
                f"ETA {eta:.1f}s",
                end=""
            )

        print()


    # --------------------------------------------------------
    # Single-sample fallback
    # --------------------------------------------------------

    if not dynamic_batch_supported:

        predictions = []

        total = len(X)

        start_time = time.time()

        # Re-create interpreter
        interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            num_threads=4
        )

        interpreter.allocate_tensors()

        input_details = (
            interpreter.get_input_details()
        )

        output_details = (
            interpreter.get_output_details()
        )

        input_index = (
            input_details[0]["index"]
        )

        output_index = (
            output_details[0]["index"]
        )

        for i in range(total):

            sample = X[
                i:i + 1
            ]

            interpreter.set_tensor(
                input_index,
                sample
            )

            interpreter.invoke()

            output = interpreter.get_tensor(
                output_index
            )

            prediction = int(
                np.argmax(
                    output[0]
                )
            )

            predictions.append(
                prediction
            )

            if (
                (i + 1) % 25 == 0
                or i == total - 1
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    (i + 1)
                    / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    total
                    - (i + 1)
                )

                eta = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                print(
                    f"\rTFLite testing "
                    f"{i+1}/{total} | "
                    f"ETA {eta:.1f}s",
                    end=""
                )

        print()


    predictions = np.asarray(
        predictions,
        dtype=np.int64
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    return (
        accuracy,
        predictions
    )


tflite_accuracy, tflite_predictions = (
    evaluate_tflite(
        TFLITE_MODEL_PATH,
        X_test,
        y_test
    )
)


# ============================================================
# 44. TFLITE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TFLITE RESULT")
print("=" * 70)

print(
    f"TFLite test accuracy: "
    f"{tflite_accuracy * 100:.2f}%"
)


print(
    f"Keras test accuracy:  "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"Accuracy difference:   "
    f"{abs(test_accuracy - tflite_accuracy) * 100:.4f}%"
)


# ============================================================
# 45. TFLITE CONFUSION MATRIX
# ============================================================

tflite_cm = confusion_matrix(
    y_test,
    tflite_predictions
)


tflite_cm_path = (
    OUTPUT_DIR
    / "tflite_confusion_matrix.png"
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    tflite_cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Greens"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "TFLite Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    tflite_cm_path,
    dpi=200
)

plt.close()


# ============================================================
# 46. DEPLOYMENT INFORMATION
# ============================================================

deployment_info = {

    "model": "BabyAudioStrongCNN",

    "tensorflow_version":
        str(tf.__version__),

    "num_classes":
        int(NUM_CLASSES),

    "classes":
        CLASS_NAMES,

    "input_shape":
        [
            1,
            MAX_FRAMES,
            NUM_MFCC,
            1
        ],

    "input_dtype":
        "float32",

    "sample_rate":
        TARGET_SAMPLE_RATE,

    "mfcc_library":
        "python_speech_features",

    "mfcc_parameters":
        {
            "numcep": NUM_MFCC,
            "winlen": WINLEN,
            "winstep": WINSTEP,
            "nfft": NFFT
        },

    "fixed_frames":
        MAX_FRAMES,

    "keras_test_accuracy":
        float(test_accuracy),

    "tflite_test_accuracy":
        float(tflite_accuracy),

    "training_time_seconds":
        float(training_time),

    "batch_size":
        BATCH_SIZE,

    "normalization_embedded_in_model":
        True,

    "pi_needs_normalization_file":
        False,

    "tflite_model":
        str(TFLITE_MODEL_PATH),

    "label_map":
        str(LABEL_MAP_PATH)
}


with open(
    INFO_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        deployment_info,
        f,
        indent=4
    )


# ============================================================
# 47. FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("TRAINING FINISHED SUCCESSFULLY")
print("=" * 70)

print()

print(
    f"Classes              : {NUM_CLASSES}"
)

print(
    f"Input                : "
    f"(500, 24, 1)"
)

print(
    f"MFCC                 : "
    f"python_speech_features"
)

print(
    f"Sample rate          : "
    f"{TARGET_SAMPLE_RATE} Hz"
)

print(
    f"Batch size           : "
    f"{BATCH_SIZE}"
)

print(
    f"Keras test accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"TFLite test accuracy : "
    f"{tflite_accuracy * 100:.2f}%"
)

print()

print(
    "BEST MODEL:"
)

print(
    BEST_MODEL_PATH.resolve()
)

print()

print(
    "TFLITE MODEL:"
)

print(
    TFLITE_MODEL_PATH.resolve()
)

print()

print(
    "LABEL MAP:"
)

print(
    LABEL_MAP_PATH.resolve()
)

print()

print(
    "CACHE:"
)

print(
    CACHE_DIR.resolve()
)

print()

print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "The Raspberry Pi must continue using the same raw"
)

print(
    "python_speech_features MFCC extraction."
)

print()

print(
    "Normalization is already embedded inside the TFLite model."
)

print()

print(
    "Do NOT normalize MFCCs again on the Raspberry Pi."
)

print()

print(
    "100% accuracy cannot be guaranteed; use the untouched"
)

print(
    "test set as the final real performance measurement."
)

print("=" * 70)