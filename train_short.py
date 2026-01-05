import os
import glob
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf

from demucs_v4_model import demucs_v4_fixed, custom_loss

SR = 44_100
CHUNK_SAMPLES = 441_000
PADDED_LEN = 441_000

INSTRUMENT_NAMES = [
    "Guitar", "Drums", "Piano", "Bass", "Strings (continued)",
    "Organ", "Synth Lead", "Synth Pad", "Chromatic Percussion",
    "Brass", "Pipe", "Reed", "Strings"
]
MODEL_KEYS = {n: f"instrument_{i+1}" for i, n in enumerate(INSTRUMENT_NAMES)}


def load_mono(path, sr=SR):
    wav, file_sr = sf.read(str(path), always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if file_sr != sr:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        except Exception:
            # As a fallback, simple crop/pad without resample
            ratio = sr / float(file_sr)
            new_len = int(len(wav) * ratio)
            wav = wav[:new_len]
    return wav


def pad_or_trim(x, tgt_len=PADDED_LEN):
    if len(x) < tgt_len:
        return np.pad(x, (0, tgt_len - len(x)))
    return x[:tgt_len]


def data_generator(root, batch_size=1):
    """Minimal generator for a quick smoke train step."""
    root = os.path.expanduser(root)
    track_dirs = [d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]
    if not track_dirs:
        raise RuntimeError(f"No track directories found under {root}")

    chunk = CHUNK_SAMPLES
    while True:
        random.shuffle(track_dirs)
        # only use a few tracks to keep it light
        for i in range(0, min(len(track_dirs), batch_size), batch_size):
            dirs = track_dirs[i:i+batch_size]

            mixes = []
            targets = {k: [] for k in MODEL_KEYS.values()}

            for d in dirs:
                mix_files = [f for f in os.listdir(d) if 'mix_chunk' in f.lower()]
                if not mix_files:
                    continue
                mix_full = load_mono(os.path.join(d, mix_files[0]))

                if len(mix_full) > chunk:
                    start = np.random.randint(0, len(mix_full) - chunk + 1)
                    mix_clip = mix_full[start:start+chunk]
                else:
                    start = 0
                    mix_clip = pad_or_trim(mix_full, chunk)

                peak = float(np.max(np.abs(mix_clip)))
                if peak < 1e-3:
                    peak = 1.0
                mix_clip = mix_clip / peak

                stem_dict = {}
                for name in INSTRUMENT_NAMES:
                    fmatch = next((f for f in os.listdir(d)
                                   if f.lower().startswith(name.lower()+"_chunk_")), None)
                    if fmatch:
                        full = load_mono(os.path.join(d, fmatch))
                        if len(full) > chunk:
                            stem = full[start:start+chunk]
                        else:
                            stem = pad_or_trim(full, chunk)
                        stem = stem / peak
                    else:
                        stem = np.zeros(chunk, dtype=np.float32)
                    stem_dict[name] = stem

                mixes.append(pad_or_trim(mix_clip, PADDED_LEN))
                for name in INSTRUMENT_NAMES:
                    targets[MODEL_KEYS[name]].append(
                        pad_or_trim(stem_dict[name], PADDED_LEN)[..., None]
                    )

            if not mixes:
                continue
            mix_batch = np.array(mixes, dtype=np.float32)[..., None]
            tgt_batch = {k: np.array(v, dtype=np.float32) for k, v in targets.items()}
            yield mix_batch, tgt_batch


def main():
    ckpt_dir = 'demucs_v4_fixed_ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPU(s)")
    else:
        strategy = tf.distribute.OneDeviceStrategy('/CPU:0')
        print("Using CPU")

    with strategy.scope():
        base_opt = tf.keras.optimizers.Adam(3e-5, clipnorm=1.0)
        from tensorflow.keras import mixed_precision
        opt = mixed_precision.LossScaleOptimizer(base_opt, dynamic=True)

        model = demucs_v4_fixed((PADDED_LEN, 1))
        model.compile(optimizer=opt, loss=custom_loss, jit_compile=False, run_eagerly=False)

    # Synthetic one-step train to validate pipeline quickly
    x = np.zeros((1, PADDED_LEN, 1), dtype=np.float32)
    y = {f"instrument_{i+1}": np.zeros((1, PADDED_LEN, 1), dtype=np.float32) for i in range(len(INSTRUMENT_NAMES))}
    print("Starting synthetic 1-step training…")
    loss = model.train_on_batch(x, y, return_dict=False)
    print("One-step loss:", float(loss) if not isinstance(loss, (list, tuple)) else loss)

    # Save a checkpoint and full model
    wpath = os.path.join(ckpt_dir, "ckpt_e01_loss999.000000.weights.h5")
    model.save_weights(wpath)
    print(f"Saved weights → {wpath}")

    out_model = 'demucs_v4_fixed_model.keras'
    model.save(out_model)
    print(f"Saved model to {out_model}")


if __name__ == '__main__':
    main()
