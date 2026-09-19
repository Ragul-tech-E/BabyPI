import os
# Hardware-style limits: CPU only, max 4 cores/threads.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import gc
import time
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

try:
    import psutil
except ImportError:
    psutil = None

MODEL = Path('Librosa_models/baby_audio_strong.tflite')
LABELS = Path('Librosa_models/label_map.json')
NORM = Path('Librosa_models/normalization.json')
MAX_CORES = 4
TFLITE_THREADS = 4
BENCHMARK_RUNS = 10
WARMUP_RUNS = 2


def limit_cpu():
    if psutil is None:
        print('WARNING: psutil not installed; CPU affinity cannot be limited.')
        print('Install: pip install psutil')
        return
    try:
        p = psutil.Process()
        old = p.cpu_affinity()
        allowed = old[:min(MAX_CORES, len(old))]
        p.cpu_affinity(allowed)
        print(f'CPU affinity: {allowed} (maximum {len(allowed)} logical CPUs)')
    except Exception as e:
        print('WARNING: CPU affinity could not be set:', e)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
        print('Runtime: tflite_runtime')
    except ImportError:
        import tensorflow as tf
        print('Runtime: TensorFlow Lite')
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(MODEL), num_threads=TFLITE_THREADS)


def prepare_audio(path, norm):
    sr_target = int(norm['sample_rate'])
    n_mfcc = int(norm['n_mfcc'])
    n_fft = int(norm['n_fft'])
    win_length = int(sr_target * float(norm['win_length']))
    hop_length = int(sr_target * float(norm['hop_length']))
    max_frames = int(norm['max_frames'])
    mean = np.asarray(norm['mean'], dtype=np.float32).reshape(1, n_mfcc)
    std = np.maximum(np.asarray(norm['std'], dtype=np.float32).reshape(1, n_mfcc), 1e-6)

    t0 = time.perf_counter()
    audio, original_sr = sf.read(str(path), dtype='float32', always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.nan_to_num(audio).astype(np.float32)
    if len(audio) == 0:
        raise ValueError('Audio file is empty.')

    if original_sr != sr_target:
        target_len = int(round(len(audio) * sr_target / float(original_sr)))
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr_target).astype(np.float32)

    load_resample_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    # EXACT preprocessing from the uploaded model code.
    audio -= np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio /= peak

    features = librosa.feature.mfcc(
        y=audio,
        sr=sr_target,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True
    ).T

    original_frames = features.shape[0]
    if original_frames < max_frames:
        features = np.pad(features, ((0, max_frames - original_frames), (0, 0)), mode='constant')
        action = f'PAD {max_frames - original_frames} frames'
    else:
        features = features[:max_frames, :]
        action = 'TRUNCATE to 500 frames' if original_frames > max_frames else 'NO padding/truncation'

    # Same external normalization as the uploaded code.
    features = ((features - mean) / std).astype(np.float32)
    features = features[..., np.newaxis][np.newaxis, ...]
    mfcc_ms = (time.perf_counter() - t1) * 1000

    return features, {
        'original_sr': int(original_sr),
        'final_sr': sr_target,
        'duration': len(audio) / sr_target,
        'frames_before': int(original_frames),
        'action': action,
        'shape': tuple(features.shape),
        'load_resample_ms': load_resample_ms,
        'mfcc_ms': mfcc_ms,
        'preprocess_ms': load_resample_ms + mfcc_ms
    }


def get_usage():
    if psutil is None:
        return None
    try:
        p = psutil.Process()
        return psutil.cpu_percent(interval=0.05), p.memory_info().rss / 1024 / 1024
    except Exception:
        return None


def predict(interpreter, x, labels):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    if tuple(inp['shape']) != tuple(x.shape):
        if len(inp['shape']) == 4 and tuple(inp['shape'][1:]) == tuple(x.shape[1:]):
            interpreter.resize_tensor_input(inp['index'], x.shape, strict=False)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
        else:
            raise ValueError(f'Model input {inp["shape"]} does not match {x.shape}')

    # Handle float or common int8/uint8 TFLite input.
    if inp['dtype'] == np.float32:
        model_x = x
    else:
        scale, zero = inp.get('quantization', (0.0, 0))
        if scale == 0:
            raise ValueError('Quantized input has zero scale.')
        model_x = x / scale + zero
        if inp['dtype'] == np.int8:
            model_x = np.clip(model_x, -128, 127).astype(np.int8)
        elif inp['dtype'] == np.uint8:
            model_x = np.clip(model_x, 0, 255).astype(np.uint8)
        else:
            model_x = model_x.astype(inp['dtype'])

    interpreter.set_tensor(inp['index'], model_x)
    t0 = time.perf_counter()
    interpreter.invoke()
    infer_ms = (time.perf_counter() - t0) * 1000
    y = interpreter.get_tensor(out['index'])[0]

    if out['dtype'] != np.float32:
        scale, zero = out.get('quantization', (0.0, 0))
        if scale:
            y = (y.astype(np.float32) - zero) * scale

    y = np.asarray(y, dtype=np.float32)
    if np.any(y < 0) or np.any(y > 1.001) or abs(float(np.sum(y)) - 1.0) > 0.05:
        z = y - np.max(y)
        e = np.exp(z)
        y = e / np.sum(e)
    else:
        y = y / max(float(np.sum(y)), 1e-12)

    idx = int(np.argmax(y))
    return idx, labels.get(idx, f'Class {idx}'), float(y[idx]), y, infer_ms


def benchmark(interpreter, x, labels):
    for _ in range(WARMUP_RUNS):
        predict(interpreter, x, labels)
    times = []
    for i in range(BENCHMARK_RUNS):
        _, _, _, _, ms = predict(interpreter, x, labels)
        times.append(ms)
        print(f'\rBenchmark {i+1:02d}/{BENCHMARK_RUNS}: {ms:.2f} ms', end='')
    print()
    return min(times), statistics.mean(times), statistics.median(times), max(times)


def main():
    limit_cpu()
    print('\n' + '=' * 70)
    print('RASPBERRY PI ZERO 2 W - LIMITED PC TEST')
    print('=' * 70)
    print('GPU                  : DISABLED')
    print('Maximum CPU cores    : 4')
    print('TFLite threads       : 4')
    print('Batch size           : 1')
    print('MFCC                 : LIBROSA (kept EXACT for current model)')
    print('Normalization        : normalization.json')
    print('=' * 70)

    if not MODEL.exists():
        raise FileNotFoundError(f'Model not found: {MODEL.resolve()}')

    labels = {int(k): str(v) for k, v in load_json(LABELS).items()}
    norm = load_json(NORM)

    interpreter = load_interpreter()
    interpreter.allocate_tensors()
    print('Input shape :', interpreter.get_input_details()[0]['shape'])
    print('Input dtype :', interpreter.get_input_details()[0]['dtype'])
    print('Output shape:', interpreter.get_output_details()[0]['shape'])
    print('Output dtype:', interpreter.get_output_details()[0]['dtype'])

    while True:
        print('\nEnter WAV path (or EXIT):')
        path_text = input('Audio path > ').strip().strip('"').strip("'")
        if path_text.lower() in ('exit', 'quit', 'q'):
            break
        path = Path(path_text)
        if not path.is_file():
            print('ERROR: file not found:', path)
            continue

        try:
            x, info = prepare_audio(path, norm)
            print('\n' + '-' * 70)
            print('AUDIO / PREPROCESSING')
            print('-' * 70)
            print('Original sample rate :', info['original_sr'], 'Hz')
            print('Final sample rate    :', info['final_sr'], 'Hz')
            print('Duration             :', f"{info['duration']:.2f} s")
            print('MFCC frames before   :', info['frames_before'])
            print('Frame operation      :', info['action'])
            print('Model input          :', info['shape'])
            print('Load/resample        :', f"{info['load_resample_ms']:.2f} ms")
            print('MFCC + normalization :', f"{info['mfcc_ms']:.2f} ms")
            print('Total preprocessing  :', f"{info['preprocess_ms']:.2f} ms")

            # One real prediction.
            idx, name, conf, probs, infer_ms = predict(interpreter, x, labels)
            print('\n' + '=' * 70)
            print('PREDICTION')
            print('=' * 70)
            print('Class      :', name)
            print('Confidence :', f'{conf * 100:.2f}%')
            print('Inference  :', f'{infer_ms:.2f} ms')
            print('\nAll classes:')
            for i in np.argsort(probs)[::-1]:
                print(f'  {labels.get(int(i), f"Class {i}"):<25} {probs[i] * 100:7.2f}%')

            total = info['preprocess_ms'] + infer_ms
            print('\nEnd-to-end:', f'{total:.2f} ms')

            usage = get_usage()
            if usage:
                print('PC CPU snapshot:', f'{usage[0]:.1f}%')
                print('Process RAM     :', f'{usage[1]:.1f} MB')

            print('\n' + '=' * 70)
            print('4-THREAD TFLITE BENCHMARK')
            print('=' * 70)
            mn, avg, med, mx = benchmark(interpreter, x, labels)
            print(f'Minimum : {mn:.2f} ms')
            print(f'Average : {avg:.2f} ms')
            print(f'Median  : {med:.2f} ms')
            print(f'Maximum : {mx:.2f} ms')

            del x
            gc.collect()

        except Exception as e:
            print('\nERROR:', e)


if __name__ == '__main__':
    main()
