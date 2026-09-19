import os
import random
import shutil

# ============================================================
# SETTINGS
# ============================================================

# Automatically use the folder where this Python script is run
SOURCE_ROOT = os.getcwd()

# Output folder
OUTPUT_ROOT = os.path.join(SOURCE_ROOT, "dataset_split")

# Train / Validation / Test percentages
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed so the same split can be reproduced
RANDOM_SEED = 42

# Supported audio files
AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma"
)

# ============================================================
# CHECK RATIOS
# ============================================================

if abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) > 0.001:
    raise ValueError("TRAIN + VALIDATION + TEST ratios must equal 1.0")

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)

# ============================================================
# FIND CONDITION FOLDERS
# ============================================================

print("=" * 60)
print("AUDIO DATASET SPLITTER")
print("=" * 60)

condition_folders = []

for item in os.listdir(SOURCE_ROOT):

    full_path = os.path.join(SOURCE_ROOT, item)

    # Ignore files
    if not os.path.isdir(full_path):
        continue

    # Ignore output folder
    if item == "dataset_split":
        continue

    condition_folders.append(item)

condition_folders.sort()

# ============================================================
# CHECK CONDITIONS
# ============================================================

if len(condition_folders) != 5:
    print()
    print("ERROR")
    print("-" * 60)
    print(f"Expected 5 condition folders.")
    print(f"Found {len(condition_folders)} folders:")
    
    for folder in condition_folders:
        print("  -", folder)

    print()
    print("Example:")
    print("  Asphyxia")
    print("  Deaf")
    print("  Hunger")
    print("  Normal")
    print("  Pain")
    
    raise SystemExit

print()
print("Found 5 conditions:")

for condition in condition_folders:
    print("  -", condition)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

splits = [
    "train",
    "validation",
    "test"
]

for split in splits:

    for condition in condition_folders:

        folder = os.path.join(
            OUTPUT_ROOT,
            split,
            condition
        )

        os.makedirs(folder, exist_ok=True)

# ============================================================
# PROCESS EACH CONDITION
# ============================================================

total_train = 0
total_validation = 0
total_test = 0

print()
print("=" * 60)
print("PROCESSING AUDIO")
print("=" * 60)

for condition in condition_folders:

    source_folder = os.path.join(
        SOURCE_ROOT,
        condition
    )

    # --------------------------------------------------------
    # FIND AUDIO FILES
    # --------------------------------------------------------

    audio_files = []

    for filename in os.listdir(source_folder):

        filepath = os.path.join(
            source_folder,
            filename
        )

        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(AUDIO_EXTENSIONS):
            audio_files.append(filepath)

    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.shuffle(audio_files)

    total_files = len(audio_files)

    if total_files == 0:
        print()
        print(f"{condition}: NO AUDIO FILES")
        continue

    # --------------------------------------------------------
    # CALCULATE SPLIT
    # --------------------------------------------------------

    train_count = int(total_files * TRAIN_RATIO)

    validation_count = int(
        total_files * VALIDATION_RATIO
    )

    test_count = (
        total_files
        - train_count
        - validation_count
    )

    train_files = audio_files[
        :train_count
    ]

    validation_files = audio_files[
        train_count:
        train_count + validation_count
    ]

    test_files = audio_files[
        train_count + validation_count:
    ]

    # --------------------------------------------------------
    # PRINT INFORMATION
    # --------------------------------------------------------

    print()
    print(f"Condition: {condition}")
    print("-" * 40)
    print(f"Total      : {total_files}")
    print(f"Train      : {len(train_files)}")
    print(f"Validation : {len(validation_files)}")
    print(f"Test       : {len(test_files)}")

    # --------------------------------------------------------
    # COPY TRAIN
    # --------------------------------------------------------

    for index, source_file in enumerate(
        train_files,
        start=1
    ):

        extension = os.path.splitext(
            source_file
        )[1].lower()

        new_filename = (
            f"{condition}_{index:04d}{extension}"
        )

        destination = os.path.join(
            OUTPUT_ROOT,
            "train",
            condition,
            new_filename
        )

        shutil.copy2(
            source_file,
            destination
        )

    # --------------------------------------------------------
    # COPY VALIDATION
    # --------------------------------------------------------

    for index, source_file in enumerate(
        validation_files,
        start=1
    ):

        extension = os.path.splitext(
            source_file
        )[1].lower()

        new_filename = (
            f"{condition}_{index:04d}{extension}"
        )

        destination = os.path.join(
            OUTPUT_ROOT,
            "validation",
            condition,
            new_filename
        )

        shutil.copy2(
            source_file,
            destination
        )

    # --------------------------------------------------------
    # COPY TEST
    # --------------------------------------------------------

    for index, source_file in enumerate(
        test_files,
        start=1
    ):

        extension = os.path.splitext(
            source_file
        )[1].lower()

        new_filename = (
            f"{condition}_{index:04d}{extension}"
        )

        destination = os.path.join(
            OUTPUT_ROOT,
            "test",
            condition,
            new_filename
        )

        shutil.copy2(
            source_file,
            destination
        )

    total_train += len(train_files)
    total_validation += len(validation_files)
    total_test += len(test_files)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("COMPLETED")
print("=" * 60)

print()
print(f"Output folder:")
print(OUTPUT_ROOT)

print()
print("TOTAL DATASET")
print("-" * 40)
print(f"Train      : {total_train}")
print(f"Validation : {total_validation}")
print(f"Test       : {total_test}")
print(
    f"Total      : "
    f"{total_train + total_validation + total_test}"
)

print()
print("Split ratios:")
print(f"Train      : {TRAIN_RATIO * 100:.0f}%")
print(f"Validation : {VALIDATION_RATIO * 100:.0f}%")
print(f"Test       : {TEST_RATIO * 100:.0f}%")

print()
print("Dataset created successfully.")
print("=" * 60)