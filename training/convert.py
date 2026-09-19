import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from python_speech_features import mfcc


# ============================================================
# SETTINGS
# ============================================================

INPUT_WAV = r"C:\Users\Raghul\Desktop\BabyPI\train_Deaf_13.wav"
OUTPUT_WAV = r"C:\Users\Raghul\Desktop\BabyPI\train_con_Deaf_13.wav"

TARGET_SAMPLE_RATE = 16000

NUMCEP = 24
WINLEN = 0.025
WINSTEP = 0.01
NFFT = 1024

TARGET_FRAMES = 500


# ============================================================
# CONVERT WAV
# ============================================================

def convert_wav(input_path, output_path):

    # --------------------------------------------------------
    # READ ORIGINAL WAV
    # --------------------------------------------------------

    audio, original_sr = sf.read(input_path)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    print("=" * 60)
    print("ORIGINAL AUDIO")
    print("=" * 60)

    print(f"Original sample rate : {original_sr} Hz")
    print(f"Original shape       : {audio.shape}")

    original_samples = len(audio)

    # --------------------------------------------------------
    # RESAMPLE 48000 -> 16000
    # --------------------------------------------------------

    if original_sr != TARGET_SAMPLE_RATE:

        audio_16k = resample_poly(
            audio,
            TARGET_SAMPLE_RATE,
            original_sr
        ).astype(np.float32)

        resampled = "YES"

    else:

        audio_16k = audio
        resampled = "NO"

    # --------------------------------------------------------
    # PRINT SAMPLE INFORMATION
    # --------------------------------------------------------

    print(f"Original samples     : {original_samples}")
    print(f"Final sample rate    : {TARGET_SAMPLE_RATE} Hz")
    print(f"Resampled            : {resampled}")

    # --------------------------------------------------------
    # WRITE NEW WAV
    #
    # 16 kHz
    # MONO
    # PCM 16-bit
    # --------------------------------------------------------

    sf.write(
        output_path,
        audio_16k,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )

    print()
    print("=" * 60)
    print("CONVERTED WAV")
    print("=" * 60)

    print(f"Output file          : {output_path}")
    print(f"Final samples        : {len(audio_16k)}")
    print(f"Final sample rate    : {TARGET_SAMPLE_RATE} Hz")

    # --------------------------------------------------------
    # READ BACK TO VERIFY
    # --------------------------------------------------------

    check_audio, check_sr = sf.read(output_path)

    if check_audio.ndim > 1:
        check_audio = np.mean(check_audio, axis=1)

    print(f"Verified sample rate : {check_sr} Hz")
    print(f"Verified shape       : {check_audio.shape}")

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    features = mfcc(
        audio_16k,
        samplerate=TARGET_SAMPLE_RATE,
        numcep=NUMCEP,
        winlen=WINLEN,
        winstep=WINSTEP,
        nfft=NFFT
    )

    mfcc_frames = features.shape[0]

    print()
    print("=" * 60)
    print("MFCC")
    print("=" * 60)

    print(f"MFCC frames          : {mfcc_frames}")

    # --------------------------------------------------------
    # PAD TO 500 FRAMES
    # --------------------------------------------------------

    if mfcc_frames < TARGET_FRAMES:

        pad_frames = TARGET_FRAMES - mfcc_frames

        features = np.pad(
            features,
            (
                (0, pad_frames),
                (0, 0)
            ),
            mode="constant",
            constant_values=0
        )

        print(f"MFCC action          : PAD {pad_frames} frames")

    elif mfcc_frames > TARGET_FRAMES:

        remove_frames = mfcc_frames - TARGET_FRAMES

        features = features[:TARGET_FRAMES]

        print(f"MFCC action          : TRUNCATE {remove_frames} frames")

    else:

        print("MFCC action          : NONE")

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    # (500, 24)
    features = features[..., np.newaxis]

    # (1, 500, 24, 1)
    model_input = features[np.newaxis, ...]

    model_input = model_input.astype(np.float32)

    print(f"Final model input    : {model_input.shape}")

    print()
    print("MFCC parameters      :")
    print(f"  numcep = {NUMCEP}")
    print(f"  winlen = {WINLEN}")
    print(f"  winstep = {WINSTEP}")
    print(f"  nfft = {NFFT}")

    print("=" * 60)

    return model_input


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    model_input = convert_wav(
        INPUT_WAV,
        OUTPUT_WAV
    )