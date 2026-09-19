import os
import glob
import hashlib
import shutil

import numpy as np
import soundfile as sf

from scipy import signal
from scipy.ndimage import gaussian_filter1d


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_FOLDER = "reference"
SOURCE_FOLDER = "source"
OUTPUT_FOLDER = "converted"

SPLITS = [
    "train",
    "validation",
    "test"
]

# Set to None to automatically detect the 5 class folders
CLASSES = None

# Expected number of phone folders
EXPECTED_PHONES = 6


# ============================================================
# AUDIO SETTINGS
# ============================================================

TARGET_SAMPLE_RATE = 16000

FFT_SIZE = 1024
HOP_SIZE = 256

SPECTRUM_SMOOTHING = 12

# Maximum spectral modification
MAX_EQ_DB = 8.0


# ============================================================
# RANDOMIZATION SETTINGS
# ============================================================

# Random loudness variation around reference
GAIN_MIN_DB = -6.0
GAIN_MAX_DB = 4.0

# Random noise SNR
SNR_MIN_DB = 12.0
SNR_MAX_DB = 30.0

# Spectral matching strength
EQ_STRENGTH_MIN = 0.50
EQ_STRENGTH_MAX = 0.90

# Small random spectral variation
EQ_RANDOM_VARIATION_DB = 1.5


# ============================================================
# ENABLE / DISABLE AUGMENTATIONS
# ============================================================

ENABLE_RANDOM_GAIN = True
ENABLE_NOISE = True
ENABLE_SPECTRAL_MATCH = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

GLOBAL_SEED = 12345


# ============================================================
# OUTPUT OPTIONS
# ============================================================

# Add "_phone1", "_phone2", etc.
ADD_PHONE_TO_FILENAME = True

# If True, remove old converted folder before starting
CLEAR_OUTPUT_FOLDER = False


# ============================================================
# BASIC UTILITIES
# ============================================================

def to_mono(audio):

    if audio.ndim == 1:
        return audio.astype(np.float32)

    return np.mean(
        audio,
        axis=1
    ).astype(np.float32)


def rms(audio):

    audio = audio.astype(np.float64)

    return float(
        np.sqrt(
            np.mean(audio ** 2) + 1e-12
        )
    )


def peak(audio):

    if len(audio) == 0:
        return 0.0

    return float(
        np.max(np.abs(audio))
    )


def db_to_linear(db):

    return 10.0 ** (
        db / 20.0
    )


def linear_to_db(value):

    return 20.0 * np.log10(
        np.maximum(
            value,
            1e-12
        )
    )


# ============================================================
# DETERMINISTIC RANDOM GENERATOR
# ============================================================

def create_rng(
        source_file,
        phone_name):

    text = (
        str(GLOBAL_SEED)
        + "_"
        + source_file
        + "_"
        + phone_name
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    seed = int(
        digest[:8],
        16
    )

    return np.random.default_rng(
        seed
    )


# ============================================================
# RESAMPLING
# ============================================================

def resample_audio(
        audio,
        old_sr,
        new_sr):

    if old_sr == new_sr:
        return audio.astype(
            np.float32
        )

    gcd = np.gcd(
        old_sr,
        new_sr
    )

    up = new_sr // gcd
    down = old_sr // gcd

    result = signal.resample_poly(
        audio,
        up,
        down
    )

    return result.astype(
        np.float32
    )


# ============================================================
# DISCOVER PHONE FOLDERS
# ============================================================

def find_phone_folders():

    if not os.path.isdir(
        REFERENCE_FOLDER
    ):
        raise RuntimeError(
            f"Missing folder: "
            f"{REFERENCE_FOLDER}"
        )

    folders = []

    for name in sorted(
        os.listdir(
            REFERENCE_FOLDER
        )
    ):

        path = os.path.join(
            REFERENCE_FOLDER,
            name
        )

        if os.path.isdir(path):

            folders.append(
                (
                    name,
                    path
                )
            )

    if len(folders) == 0:

        raise RuntimeError(
            "No phone folders found inside "
            "reference folder."
        )

    if len(folders) != EXPECTED_PHONES:

        raise RuntimeError(
            f"Expected {EXPECTED_PHONES} "
            f"phone folders, but found "
            f"{len(folders)}.\n\n"
            f"Expected structure:\n"
            f"reference/phone1\n"
            f"reference/phone2\n"
            f"reference/phone3\n"
            f"reference/phone4\n"
            f"reference/phone5"
        )

    return folders


# ============================================================
# FIND REFERENCE WAV FILES FOR PHONE
# ============================================================

def find_phone_reference_files(
        phone_folder):

    files = glob.glob(
        os.path.join(
            phone_folder,
            "**",
            "*.wav"
        ),
        recursive=True
    )

    return sorted(files)


# ============================================================
# FIND SOURCE DATA
# ============================================================

def find_source_files():

    files = []

    for split in SPLITS:

        split_folder = os.path.join(
            SOURCE_FOLDER,
            split
        )

        if not os.path.isdir(
            split_folder
        ):

            raise RuntimeError(
                f"Missing source split: "
                f"{split_folder}"
            )

        # ----------------------------------------------------
        # Detect class folders automatically
        # ----------------------------------------------------

        if CLASSES is None:

            class_names = []

            for name in sorted(
                os.listdir(
                    split_folder
                )
            ):

                path = os.path.join(
                    split_folder,
                    name
                )

                if os.path.isdir(path):

                    class_names.append(
                        name
                    )

        else:

            class_names = CLASSES

        if len(class_names) != 5:

            raise RuntimeError(
                f"{split} must contain "
                f"exactly 5 class folders. "
                f"Found: {class_names}"
            )

        # ----------------------------------------------------
        # Find WAV files
        # ----------------------------------------------------

        for class_name in class_names:

            class_folder = os.path.join(
                split_folder,
                class_name
            )

            if not os.path.isdir(
                class_folder
            ):

                raise RuntimeError(
                    f"Missing class folder: "
                    f"{class_folder}"
                )

            class_files = glob.glob(
                os.path.join(
                    class_folder,
                    "**",
                    "*.wav"
                ),
                recursive=True
            )

            for filename in sorted(
                class_files
            ):

                files.append(
                    (
                        split,
                        class_name,
                        filename
                    )
                )

    return files


# ============================================================
# STFT AVERAGE SPECTRUM
# ============================================================

def calculate_average_spectrum(
        audio,
        sample_rate):

    audio = audio.astype(
        np.float32
    )

    if len(audio) < FFT_SIZE:

        padded = np.zeros(
            FFT_SIZE,
            dtype=np.float32
        )

        padded[
            :len(audio)
        ] = audio

        audio = padded

    frequencies, times, Zxx = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE,
        boundary="zeros",
        padded=True
    )

    magnitude = np.abs(Zxx)

    spectrum = np.mean(
        magnitude,
        axis=1
    )

    return (
        frequencies,
        spectrum
    )


# ============================================================
# EXTRACT LOW-ENERGY NOISE SEGMENTS
# ============================================================

def extract_noise_segments(
        audio):

    segments = []

    frame_length = int(
        TARGET_SAMPLE_RATE * 0.20
    )

    hop = int(
        TARGET_SAMPLE_RATE * 0.10
    )

    if len(audio) < frame_length:

        return segments

    frame_rms = []
    positions = []

    for start in range(
        0,
        len(audio) -
        frame_length + 1,
        hop
    ):

        frame = audio[
            start:
            start + frame_length
        ]

        frame_rms.append(
            rms(frame)
        )

        positions.append(
            start
        )

    if len(frame_rms) == 0:

        return segments

    frame_rms = np.array(
        frame_rms
    )

    # Lowest 20% energy
    threshold = np.percentile(
        frame_rms,
        20
    )

    for start, value in zip(
        positions,
        frame_rms
    ):

        if value > threshold:
            continue

        segment = audio[
            start:
            start + frame_length
        ]

        if len(segment) == frame_length:

            # Reject complete silence
            if rms(segment) > 1e-7:

                segments.append(
                    segment.copy()
                )

    return segments


# ============================================================
# ANALYZE ONE PHONE
# ============================================================

def analyze_phone(
        phone_name,
        phone_folder):

    reference_files = (
        find_phone_reference_files(
            phone_folder
        )
    )

    if len(reference_files) == 0:

        raise RuntimeError(
            f"No WAV files found for "
            f"{phone_name}"
        )

    print()
    print("=" * 70)
    print(
        f"ANALYZING {phone_name}"
    )
    print("=" * 70)

    print(
        f"Reference recordings: "
        f"{len(reference_files)}"
    )

    rms_values = []
    spectra = []
    noise_segments = []

    # --------------------------------------------------------
    # Analyze all reference files
    # --------------------------------------------------------

    for index, filename in enumerate(
        reference_files,
        1
    ):

        print(
            f"[{index}/{len(reference_files)}] "
            f"{os.path.basename(filename)}"
        )

        audio, sr = sf.read(
            filename,
            always_2d=False
        )

        audio = to_mono(
            audio
        )

        # ----------------------------------------------------
        # RMS
        # ----------------------------------------------------

        rms_values.append(
            rms(audio)
        )

        # ----------------------------------------------------
        # Resample
        # ----------------------------------------------------

        audio = resample_audio(
            audio,
            sr,
            TARGET_SAMPLE_RATE
        )

        # ----------------------------------------------------
        # Remove DC
        # ----------------------------------------------------

        audio = (
            audio -
            np.mean(audio)
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Spectrum
        # ----------------------------------------------------

        freqs, spectrum = (
            calculate_average_spectrum(
                audio,
                TARGET_SAMPLE_RATE
            )
        )

        spectrum_db = linear_to_db(
            spectrum
        )

        # Remove overall loudness
        spectrum_db -= np.mean(
            spectrum_db
        )

        # Smooth
        spectrum_db = (
            gaussian_filter1d(
                spectrum_db,
                SPECTRUM_SMOOTHING
            )
        )

        spectra.append(
            spectrum_db
        )

        # ----------------------------------------------------
        # Noise
        # ----------------------------------------------------

        extracted = (
            extract_noise_segments(
                audio
            )
        )

        noise_segments.extend(
            extracted
        )

    # ========================================================
    # REFERENCE SPECTRUM
    # ========================================================

    reference_spectrum = np.median(
        np.array(
            spectra
        ),
        axis=0
    )

    reference_spectrum -= np.mean(
        reference_spectrum
    )

    # ========================================================
    # REFERENCE RMS
    # ========================================================

    target_rms = float(
        np.median(
            rms_values
        )
    )

    # ========================================================
    # NOISE
    # ========================================================

    if len(noise_segments) > 0:

        noise_rms_values = [
            rms(x)
            for x in noise_segments
        ]

        reference_noise_rms = float(
            np.median(
                noise_rms_values
            )
        )

    else:

        reference_noise_rms = 0.0

    # ========================================================
    # PROFILE
    # ========================================================

    profile = {

        "phone":
            phone_name,

        "sample_rate":
            TARGET_SAMPLE_RATE,

        "rms":
            target_rms,

        "spectrum":
            reference_spectrum,

        "frequencies":
            freqs,

        "noise_segments":
            noise_segments,

        "noise_rms":
            reference_noise_rms
    }

    print()
    print(
        f"{phone_name} PROFILE"
    )

    print(
        f"Recordings : "
        f"{len(reference_files)}"
    )

    print(
        f"Target RMS : "
        f"{target_rms:.6f}"
    )

    print(
        f"Noise segments : "
        f"{len(noise_segments)}"
    )

    print(
        f"Noise RMS : "
        f"{reference_noise_rms:.6f}"
    )

    return profile


# ============================================================
# ADD RANDOM PHONE NOISE
# ============================================================

def add_random_reference_noise(
        audio,
        profile,
        rng):

    if not ENABLE_NOISE:

        return audio

    noise_segments = profile[
        "noise_segments"
    ]

    if len(noise_segments) == 0:

        return audio

    # --------------------------------------------------------
    # Random noise segment
    # --------------------------------------------------------

    index = rng.integers(
        0,
        len(noise_segments)
    )

    noise = noise_segments[
        index
    ].copy()

    # --------------------------------------------------------
    # Repeat to duration
    # --------------------------------------------------------

    repetitions = int(
        np.ceil(
            len(audio) /
            len(noise)
        )
    )

    noise = np.tile(
        noise,
        repetitions
    )

    noise = noise[
        :len(audio)
    ]

    # --------------------------------------------------------
    # Random shift
    # --------------------------------------------------------

    if len(noise) > 1:

        shift = rng.integers(
            0,
            len(noise)
        )

        noise = np.roll(
            noise,
            shift
        )

    # --------------------------------------------------------
    # Normalize noise
    # --------------------------------------------------------

    noise_rms = rms(
        noise
    )

    if noise_rms <= 1e-10:

        return audio

    noise /= noise_rms

    # --------------------------------------------------------
    # Random SNR
    # --------------------------------------------------------

    snr_db = rng.uniform(
        SNR_MIN_DB,
        SNR_MAX_DB
    )

    signal_rms = rms(
        audio
    )

    if signal_rms <= 1e-10:

        return audio

    desired_noise_rms = (
        signal_rms /
        db_to_linear(
            snr_db
        )
    )

    noise *= desired_noise_rms

    return (
        audio + noise
    ).astype(
        np.float32
    )


# ============================================================
# APPLY PHONE SPECTRAL CHARACTER
# ============================================================

def apply_spectral_coloration(
        audio,
        sample_rate,
        profile,
        rng):

    if not ENABLE_SPECTRAL_MATCH:

        return audio

    if len(audio) < FFT_SIZE:

        return audio

    reference_spectrum = profile[
        "spectrum"
    ]

    reference_frequencies = profile[
        "frequencies"
    ]

    # --------------------------------------------------------
    # Source spectrum
    # --------------------------------------------------------

    freqs, source_spectrum = (
        calculate_average_spectrum(
            audio,
            sample_rate
        )
    )

    source_db = linear_to_db(
        source_spectrum
    )

    source_db -= np.mean(
        source_db
    )

    # --------------------------------------------------------
    # Reference spectrum
    # --------------------------------------------------------

    reference_db = np.interp(
        freqs,
        reference_frequencies,
        reference_spectrum
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    difference_db = (
        reference_db -
        source_db
    )

    difference_db = (
        gaussian_filter1d(
            difference_db,
            SPECTRUM_SMOOTHING
        )
    )

    # --------------------------------------------------------
    # Random strength
    # --------------------------------------------------------

    strength = rng.uniform(
        EQ_STRENGTH_MIN,
        EQ_STRENGTH_MAX
    )

    difference_db *= strength

    # --------------------------------------------------------
    # Small random variation
    # --------------------------------------------------------

    variation = rng.normal(
        0,
        EQ_RANDOM_VARIATION_DB,
        len(difference_db)
    )

    variation = (
        gaussian_filter1d(
            variation,
            SPECTRUM_SMOOTHING * 2
        )
    )

    difference_db += variation

    # --------------------------------------------------------
    # Limit EQ
    # --------------------------------------------------------

    difference_db = np.clip(
        difference_db,
        -MAX_EQ_DB,
        MAX_EQ_DB
    )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    frequencies, times, Zxx = (
        signal.stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=FFT_SIZE,
            noverlap=FFT_SIZE - HOP_SIZE,
            boundary="zeros",
            padded=True
        )
    )

    gain_db = np.interp(
        frequencies,
        freqs,
        difference_db
    )

    gain = db_to_linear(
        gain_db
    )

    Zxx *= gain[:, None]

    _, output = signal.istft(
        Zxx,
        fs=sample_rate,
        window="hann",
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE,
        input_onesided=True,
        boundary=True
    )

    output = output[
        :len(audio)
    ]

    return output.astype(
        np.float32
    )


# ============================================================
# RANDOMIZE LOUDNESS
# ============================================================

def randomize_loudness(
        audio,
        profile,
        rng):

    if not ENABLE_RANDOM_GAIN:

        return audio

    current_rms = rms(
        audio
    )

    if current_rms <= 1e-10:

        return audio

    target_rms = profile[
        "rms"
    ]

    base_gain = (
        target_rms /
        current_rms
    )

    # Random real-world volume variation
    random_db = rng.uniform(
        GAIN_MIN_DB,
        GAIN_MAX_DB
    )

    random_gain = (
        db_to_linear(
            random_db
        )
    )

    gain = (
        base_gain *
        random_gain
    )

    gain = np.clip(
        gain,
        0.1,
        10.0
    )

    return (
        audio * gain
    ).astype(
        np.float32
    )


# ============================================================
# SOFT CLIPPING / LIMITING
# ============================================================

def prevent_clipping(
        audio):

    maximum = np.max(
        np.abs(audio)
    )

    if maximum <= 0.98:

        return audio

    # Smooth limiting
    return np.tanh(
        audio / maximum
    ).astype(
        np.float32
    )


# ============================================================
# CREATE PHONE-SPECIFIC FILENAME
# ============================================================

def make_output_filename(
        source_file,
        phone_name):

    filename = os.path.basename(
        source_file
    )

    name, extension = (
        os.path.splitext(
            filename
        )
    )

    if ADD_PHONE_TO_FILENAME:

        name = (
            f"{name}_{phone_name}"
        )

    return (
        name +
        extension
    )


# ============================================================
# PROCESS ONE SOURCE AUDIO THROUGH ONE PHONE
# ============================================================

def process_file(
        source_file,
        split,
        class_name,
        phone_name,
        profile):

    rng = create_rng(
        source_file,
        phone_name
    )

    print(
        f"[{split}] "
        f"[{class_name}] "
        f"[{phone_name}] "
        f"{os.path.basename(source_file)}"
    )

    # --------------------------------------------------------
    # READ SOURCE
    # --------------------------------------------------------

    audio, source_sr = sf.read(
        source_file,
        always_2d=False
    )

    # --------------------------------------------------------
    # MONO
    # --------------------------------------------------------

    audio = to_mono(
        audio
    )

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    audio = resample_audio(
        audio,
        source_sr,
        profile[
            "sample_rate"
        ]
    )

    # --------------------------------------------------------
    # REMOVE DC OFFSET
    # --------------------------------------------------------

    audio = (
        audio -
        np.mean(audio)
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # PHONE SPECTRAL CHARACTER
    # --------------------------------------------------------

    audio = apply_spectral_coloration(
        audio,
        profile[
            "sample_rate"
        ],
        profile,
        rng
    )

    # --------------------------------------------------------
    # PHONE LOUDNESS
    # --------------------------------------------------------

    audio = randomize_loudness(
        audio,
        profile,
        rng
    )

    # --------------------------------------------------------
    # PHONE NOISE
    # --------------------------------------------------------

    audio = add_random_reference_noise(
        audio,
        profile,
        rng
    )

    # --------------------------------------------------------
    # PREVENT CLIPPING
    # --------------------------------------------------------

    audio = prevent_clipping(
        audio
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_directory = os.path.join(
        OUTPUT_FOLDER,
        split,
        class_name
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OUTPUT NAME
    # --------------------------------------------------------

    output_filename = (
        make_output_filename(
            source_file,
            phone_name
        )
    )

    output_path = os.path.join(
        output_directory,
        output_filename
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    sf.write(
        output_path,
        audio,
        profile[
            "sample_rate"
        ],
        subtype="PCM_16"
    )

    return output_path


# ============================================================
# VERIFY SOURCE DATASET
# ============================================================

def verify_source_dataset(
        source_files):

    print()
    print("=" * 70)
    print("SOURCE DATASET SUMMARY")
    print("=" * 70)

    summary = {}

    for split in SPLITS:

        summary[split] = {}

        for (
            s,
            class_name,
            filename
        ) in source_files:

            if s != split:
                continue

            if class_name not in summary[
                split
            ]:

                summary[
                    split
                ][
                    class_name
                ] = 0

            summary[
                split
            ][
                class_name
            ] += 1

    for split in SPLITS:

        print()
        print(
            f"{split.upper()}"
        )

        for class_name in sorted(
            summary[split]
        ):

            print(
                f"  {class_name:20s}"
                f"{summary[split][class_name]:6d}"
            )

    return summary


# ============================================================
# VERIFY REFERENCE DATASET
# ============================================================

def verify_reference_dataset(
        phone_folders):

    print()
    print("=" * 70)
    print("REFERENCE DATASET")
    print("=" * 70)

    for phone_name, phone_folder in phone_folders:

        files = (
            find_phone_reference_files(
                phone_folder
            )
        )

        print(
            f"{phone_name:15s}"
            f"{len(files):6d} recordings"
        )

        if len(files) == 0:

            raise RuntimeError(
                f"{phone_name} has no "
                f"WAV files."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("5-PHONE AUDIO RECORDING SIMULATOR")
    print("=" * 70)

    # ========================================================
    # CLEAR OUTPUT
    # ========================================================

    if CLEAR_OUTPUT_FOLDER:

        if os.path.exists(
            OUTPUT_FOLDER
        ):

            print(
                f"Removing old output: "
                f"{OUTPUT_FOLDER}"
            )

            shutil.rmtree(
                OUTPUT_FOLDER
            )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # ========================================================
    # FIND PHONES
    # ========================================================

    phone_folders = (
        find_phone_folders()
    )

    print()
    print(
        "Phones detected:"
    )

    for phone_name, phone_folder in phone_folders:

        print(
            f"  {phone_name} -> "
            f"{phone_folder}"
        )

    verify_reference_dataset(
        phone_folders
    )

    # ========================================================
    # BUILD PHONE PROFILES
    # ========================================================

    profiles = {}

    for phone_name, phone_folder in phone_folders:

        profiles[
            phone_name
        ] = analyze_phone(
            phone_name,
            phone_folder
        )

    # ========================================================
    # FIND SOURCE FILES
    # ========================================================

    source_files = (
        find_source_files()
    )

    if len(source_files) == 0:

        raise RuntimeError(
            "No source WAV files found."
        )

    summary = (
        verify_source_dataset(
            source_files
        )
    )

    # ========================================================
    # CALCULATE EXPECTED OUTPUT
    # ========================================================

    expected_output = (
        len(source_files) *
        len(phone_folders)
    )

    print()
    print("=" * 70)
    print("CONVERSION PLAN")
    print("=" * 70)

    print(
        f"Source files : "
        f"{len(source_files)}"
    )

    print(
        f"Phones       : "
        f"{len(phone_folders)}"
    )

    print(
        f"Expected synthetic files : "
        f"{expected_output}"
    )

    # ========================================================
    # PROCESS
    # ========================================================

    converted_count = 0

    print()
    print("=" * 70)
    print("STARTING CONVERSION")
    print("=" * 70)

    for (
        split,
        class_name,
        source_file
    ) in source_files:

        # ----------------------------------------------------
        # Run same source through every phone
        # ----------------------------------------------------

        for (
            phone_name,
            phone_folder
        ) in phone_folders:

            try:

                process_file(
                    source_file,
                    split,
                    class_name,
                    phone_name,
                    profiles[
                        phone_name
                    ]
                )

                converted_count += 1

            except Exception as error:

                print()
                print(
                    "ERROR processing:"
                )

                print(
                    source_file
                )

                print(
                    phone_name
                )

                print(
                    error
                )

                raise

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Source files      : "
        f"{len(source_files)}"
    )

    print(
        f"Phone profiles     : "
        f"{len(phone_folders)}"
    )

    print(
        f"Expected outputs   : "
        f"{expected_output}"
    )

    print(
        f"Actual outputs     : "
        f"{converted_count}"
    )

    print()
    print(
        f"Output directory:"
    )

    print(
        os.path.abspath(
            OUTPUT_FOLDER
        )
    )

    # ========================================================
    # CHECK
    # ========================================================

    if converted_count == expected_output:

        print()
        print(
            "STATUS: SUCCESS"
        )

    else:

        print()
        print(
            "STATUS: WARNING"
        )

        print(
            "Expected and actual "
            "file counts differ."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()