import os
import shutil
import numpy as np
import librosa
import soundfile as sf


# ============================================================
# SETTINGS
# ============================================================

# The directory where this Python script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# DESTINATION FOLDER
# ------------------------------------------------------------
# IMPORTANT:
# Change this only if your combined dataset has another name.
#
# Example:
# Combined/
# Dataset/
# Final/
# Merged/
#
DESTINATION_FOLDER_NAME = "Combined"


# ------------------------------------------------------------
# AUDIO SETTINGS
# ------------------------------------------------------------

# Silence removal sensitivity.
#
# Higher value:
#   removes more quiet audio
#
# Lower value:
#   keeps more quiet audio
#
# 30 is a good starting point for recorded baby audio.
SILENCE_TOP_DB = 30


# Minimum audio length after silence removal
# in milliseconds.
#
# Prevents extremely tiny audio files.
MIN_AUDIO_LENGTH_MS = 100


# Supported audio formats
AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma"
)


# Main folders
SPLITS = [
    "train",
    "validation",
    "test"
]


# ============================================================
# FIND PHONE FOLDERS
# ============================================================

def find_phone_folders():

    phone_folders = []

    for item in os.listdir(CURRENT_DIR):

        full_path = os.path.join(
            CURRENT_DIR,
            item
        )

        if not os.path.isdir(full_path):
            continue

        # Destination is NOT a phone folder
        if item.lower() == DESTINATION_FOLDER_NAME.lower():
            continue

        # Check if this looks like Phone1, Phone2, etc.
        if item.lower().startswith("phone"):

            phone_folders.append(full_path)

    # Sort naturally
    phone_folders.sort(
        key=lambda x: (
            ''.join(
                c for c in os.path.basename(x)
                if c.isdigit()
            ) or "999999"
        )
    )

    return phone_folders


# ============================================================
# FIND CONDITION FOLDERS
# ============================================================

def find_condition_folders(phone_folder):

    conditions = []

    for item in os.listdir(phone_folder):

        full_path = os.path.join(
            phone_folder,
            item
        )

        if not os.path.isdir(full_path):
            continue

        conditions.append(item)

    conditions.sort()

    return conditions


# ============================================================
# AUDIO FILE SEARCH
# ============================================================

def find_audio_files(folder):

    audio_files = []

    if not os.path.exists(folder):
        return audio_files

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(
                AUDIO_EXTENSIONS
            ):

                full_path = os.path.join(
                    root,
                    filename
                )

                audio_files.append(full_path)

    audio_files.sort()

    return audio_files


# ============================================================
# REMOVE SILENCE
# ============================================================

def remove_beginning_and_ending_silence(
    input_file
):

    try:

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        audio, sample_rate = librosa.load(
            input_file,
            sr=None,
            mono=True
        )

        # Empty file
        if len(audio) == 0:
            return None

        # ----------------------------------------------------
        # NORMALIZE TYPE
        # ----------------------------------------------------

        audio = audio.astype(
            np.float32
        )

        # ----------------------------------------------------
        # REMOVE LEADING / TRAILING SILENCE
        # ----------------------------------------------------

        trimmed_audio, trim_indices = librosa.effects.trim(
            audio,
            top_db=SILENCE_TOP_DB
        )

        # ----------------------------------------------------
        # CHECK RESULT
        # ----------------------------------------------------

        if len(trimmed_audio) == 0:
            return None

        # Minimum duration check
        duration_ms = (
            len(trimmed_audio)
            / sample_rate
            * 1000
        )

        if duration_ms < MIN_AUDIO_LENGTH_MS:
            return None

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return trimmed_audio, sample_rate

    except Exception as e:

        print()
        print("ERROR READING:")
        print(input_file)
        print("Reason:", e)

        return None


# ============================================================
# CREATE UNIQUE FILENAME
# ============================================================

def get_next_filename(
    destination_folder,
    condition,
    phone_name
):

    prefix = (
        f"{condition}_"
        f"{phone_name}_"
    )

    existing_numbers = []

    if os.path.exists(destination_folder):

        for filename in os.listdir(
            destination_folder
        ):

            if not filename.lower().endswith(".wav"):
                continue

            if not filename.startswith(prefix):
                continue

            # Example:
            # Hunger_Phone1_0001.wav

            name_without_ext = os.path.splitext(
                filename
            )[0]

            parts = name_without_ext.split("_")

            if len(parts) < 3:
                continue

            try:

                number = int(parts[-1])

                existing_numbers.append(
                    number
                )

            except ValueError:
                pass

    if existing_numbers:

        next_number = max(
            existing_numbers
        ) + 1

    else:

        next_number = 1

    return (
        f"{prefix}"
        f"{next_number:04d}.wav"
    )


# ============================================================
# PROCESS ONE AUDIO
# ============================================================

def process_audio(
    source_file,
    destination_folder,
    condition,
    phone_name
):

    result = remove_beginning_and_ending_silence(
        source_file
    )

    if result is None:

        print(
            "SKIPPED:",
            os.path.basename(source_file)
        )

        return False

    trimmed_audio, sample_rate = result

    # --------------------------------------------------------
    # CREATE UNIQUE NAME
    # --------------------------------------------------------

    filename = get_next_filename(
        destination_folder,
        condition,
        phone_name
    )

    destination_file = os.path.join(
        destination_folder,
        filename
    )

    # --------------------------------------------------------
    # SAVE WAV
    # --------------------------------------------------------

    sf.write(
        destination_file,
        trimmed_audio,
        sample_rate,
        subtype="PCM_16"
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    duration = (
        len(trimmed_audio)
        / sample_rate
    )

    print(
        f"  {phone_name} -> "
        f"{filename} "
        f"({duration:.2f}s)"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("5-PHONE AUDIO MERGER + SILENCE REMOVER")
    print("=" * 70)

    # --------------------------------------------------------
    # FIND PHONES
    # --------------------------------------------------------

    phone_folders = find_phone_folders()

    if not phone_folders:

        print()
        print("ERROR: No Phone folders found.")
        print()
        print("Expected:")
        print("Phone1")
        print("Phone2")
        print("Phone3")
        print("Phone4")
        print("Phone5")

        return

    print()
    print("Detected phone folders:")

    for folder in phone_folders:

        print(
            "  -",
            os.path.basename(folder)
        )

    print()
    print(
        f"Total phones found: "
        f"{len(phone_folders)}"
    )

    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    destination_root = os.path.join(
        CURRENT_DIR,
        DESTINATION_FOLDER_NAME
    )

    if not os.path.exists(
        destination_root
    ):

        print()
        print(
            "ERROR: Destination folder does not exist:"
        )

        print(destination_root)

        print()
        print(
            "Create the destination folder first."
        )

        return

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    total_found = 0
    total_processed = 0
    total_skipped = 0

    # --------------------------------------------------------
    # PROCESS PHONES
    # --------------------------------------------------------

    for phone_folder in phone_folders:

        phone_name = os.path.basename(
            phone_folder
        )

        print()
        print("=" * 70)
        print(f"PROCESSING {phone_name}")
        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN / VALIDATION / TEST
        # ----------------------------------------------------

        for split in SPLITS:

            source_split = os.path.join(
                phone_folder,
                split
            )

            destination_split = os.path.join(
                destination_root,
                split
            )

            if not os.path.exists(
                source_split
            ):

                print()
                print(
                    f"WARNING: "
                    f"{phone_name}\\{split} "
                    f"does not exist."
                )

                continue

            # ------------------------------------------------
            # FIND CONDITION FOLDERS
            # ------------------------------------------------

            conditions = find_condition_folders(
                source_split
            )

            print()
            print(
                f"[{phone_name}] "
                f"{split}"
            )

            # ------------------------------------------------
            # PROCESS CONDITIONS
            # ------------------------------------------------

            for condition in conditions:

                source_condition = os.path.join(
                    source_split,
                    condition
                )

                destination_condition = os.path.join(
                    destination_split,
                    condition
                )

                # Create destination condition folder
                os.makedirs(
                    destination_condition,
                    exist_ok=True
                )

                # --------------------------------------------
                # FIND AUDIO
                # --------------------------------------------

                audio_files = find_audio_files(
                    source_condition
                )

                print()
                print(
                    f"  Condition: {condition}"
                )

                print(
                    f"  Audio files: "
                    f"{len(audio_files)}"
                )

                total_found += len(
                    audio_files
                )

                # --------------------------------------------
                # PROCESS AUDIO
                # --------------------------------------------

                for audio_file in audio_files:

                    success = process_audio(
                        audio_file,
                        destination_condition,
                        condition,
                        phone_name
                    )

                    if success:

                        total_processed += 1

                    else:

                        total_skipped += 1

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Total audio found    : "
        f"{total_found}"
    )

    print(
        f"Successfully processed: "
        f"{total_processed}"
    )

    print(
        f"Skipped               : "
        f"{total_skipped}"
    )

    print()
    print("Destination:")
    print(destination_root)

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()