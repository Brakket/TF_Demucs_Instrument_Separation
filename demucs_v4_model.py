"""
TensorFlow Hybrid Demucs V4 Model for Music Source Separation

A TensorFlow/Keras implementation of the Hybrid Demucs architecture for separating
mixed audio into 13 individual instrument stems. Combines time-domain and frequency-domain
processing with cross-attention fusion for high-quality source separation.

Instruments: Guitar, Drums, Piano, Bass, Strings (continued), Organ, Synth Lead,
             Synth Pad, Chromatic Percussion, Brass, Pipe, Reed, Strings

Reference: Defossez, A. "Hybrid Spectrogram and Waveform Source Separation" (2021)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import mixed_precision

# ──────────────────────────────────────────────────────────────
# 1)  Mixed precision policy – set *before* any layer is built
# ──────────────────────────────────────────────────────────────
# Prefer BF16 on H100 for better numeric stability than FP16
mixed_precision.set_global_policy("mixed_bfloat16")

# ==================================================================
#  Small utility layers that replace previously‑unsafe `Lambda`s
# ==================================================================

@tf.keras.utils.register_keras_serializable()
class ExpandDims(layers.Layer):
    """Adds a singleton dim. Equivalent to `tf.expand_dims(x, axis)`."""

    def __init__(self, axis=-1, **kw):
        super().__init__(**kw)
        self.axis = axis

    def call(self, x):
        return tf.expand_dims(x, self.axis)

    def compute_output_shape(self, s):
        # Insert a 1 at the desired axis position
        if self.axis < 0:
            axis = len(s) + 1 + self.axis  # normalise negative indices
        else:
            axis = self.axis
        return s[:axis] + (1,) + s[axis:]

    def get_config(self):
        cfg = super().get_config()
        cfg.update(axis=self.axis)
        return cfg


@tf.keras.utils.register_keras_serializable()
class ReduceMean(layers.Layer):
    """`tf.reduce_mean(x, axis)` but serialisable."""

    def __init__(self, axis, keepdims=False, **kw):
        super().__init__(**kw)
        self.axis = axis
        self.keep = keepdims

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis, keepdims=self.keep)

    def compute_output_shape(self, s):
        if self.keep:
            return s
        # Remove the reduced axis(es).  Make sure axis is a tuple.
        axes = self.axis if isinstance(self.axis, (list, tuple)) else (self.axis,)
        axes = tuple(a if a >= 0 else len(s) + a for a in axes)
        return tuple(d for i, d in enumerate(s) if i not in axes)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(axis=self.axis, keepdims=self.keep)
        return cfg


# ==================================================================
#  Convolution & Attention helpers
# ==================================================================

def conv1d_block(inputs, filters, kernel_size, strides=1, activation="relu", padding="same", use_gelu=False):
    """Conv1D -> BatchNorm -> Activation block."""
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    # BatchNorm in float32 for mixed precision stability
    x = layers.BatchNormalization(synchronized=True, dtype="float32")(x)
    x = keras.activations.gelu(x) if use_gelu else layers.Activation(activation)(x)
    return x


def light_conv_block(inputs, filters, kernel_size=3, dilation_rate=1, use_gelu=False):
    """Depthwise-separable convolution block for parameter-efficient processing."""
    x = layers.Conv1D(filters, kernel_size, padding="same", groups=filters, dilation_rate=dilation_rate)(inputs)
    x = layers.BatchNormalization(synchronized=True, dtype="float32")(x)
    x = keras.activations.gelu(x) if use_gelu else layers.ReLU()(x)

    x = layers.Conv1D(filters, 1, padding="same")(x)
    x = layers.BatchNormalization(synchronized=True, dtype="float32")(x)
    x = keras.activations.gelu(x) if use_gelu else layers.ReLU()(x)
    return x


# ──────────────────────────────────────────────────────────────
#  Local self‑attention with windowing (serialisable)
# ──────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable()
class LocalSelfAttention(layers.Layer):
    """Self‑attention applied independently inside fixed‑size windows."""

    def __init__(self, heads=8, key_dim=32, window_size=1024, **kw):
        super().__init__(**kw)
        self.heads = heads
        self.kdim = key_dim
        self.W = window_size
        # Compute attention in float32 to avoid fp16 overflow in softmax logits
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype="float32")
        self.mha = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=0.0, dtype="float32")

    def call(self, x):
        # shape: (B, T, C)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        C = tf.shape(x)[2]
        W = self.W

        # Normalize then run attention in float32
        z = self.norm(x)
        z = tf.cast(z, tf.float32)

        pad = (-T) % W  # right‑pad so that T is a multiple of W
        z_pad = tf.pad(z, [[0, 0], [0, pad], [0, 0]])  # (B, T+pad, C)
        n_chunks = tf.shape(z_pad)[1] // W
        z_chunks = tf.reshape(z_pad, [B * n_chunks, W, C])
        att = self.mha(z_chunks, z_chunks)
        att = tf.reshape(att, [B, n_chunks * W, C])
        att = att[:, :T, :]  # strip the padding

        # Residual connection and cast back to policy dtype
        att = att + tf.cast(x, tf.float32)
        return tf.cast(att, x.dtype)

    # Make it serialisable
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(heads=self.heads, key_dim=self.kdim, window_size=self.W))
        return cfg


# ==================================================================
#  Spectrogram layers (STFT / iSTFT)
# ==================================================================

@tf.keras.utils.register_keras_serializable()
class STFT(layers.Layer):
    """Fixed‑parameter short‑time Fourier transform."""

    def __init__(self, frame_length=4096, frame_step=1024, **kw):
        super().__init__(**kw)
        self.FL = frame_length
        self.FS = frame_step

    def call(self, x):
        x = tf.squeeze(x, -1)  # (B, samples)
        stft = tf.signal.stft(tf.cast(x, tf.float32), frame_length=self.FL, frame_step=self.FS, window_fn=tf.signal.hann_window)
        mag = tf.abs(stft)
        pha = tf.math.angle(stft)
        return tf.stack([mag, pha], axis=-1)  # (B, frames, freq, 2)

    def compute_output_shape(self, s):
        B, N, _ = s
        frames = 1 + (N - self.FL) // self.FS
        freq = self.FL // 2 + 1
        return (B, frames, freq, 2)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(frame_length=self.FL, frame_step=self.FS)
        return cfg


@tf.keras.utils.register_keras_serializable()
class InverseSTFT(layers.Layer):
    """Inverse STFT that zero‑pads the spectrogram to reach `target_len`."""

    def __init__(self, frame_length=4096, frame_step=1024, target_len=441_000, **kw):
        super().__init__(**kw)
        self.FL = frame_length
        self.FS = frame_step
        self.Lout = target_len

    def call(self, spec_mag_phase):
        mag = tf.cast(spec_mag_phase[..., 0], tf.float32)
        pha = tf.cast(spec_mag_phase[..., 1], tf.float32)
        Z = tf.complex(mag * tf.math.cos(pha), mag * tf.math.sin(pha))

        n_frames = tf.shape(Z)[1]
        need_extra = self.Lout - (self.FL + (n_frames - 1) * self.FS)
        extra = tf.maximum(tf.cast(tf.math.ceil(need_extra / self.FS), tf.int32), 0)

        pad = tf.zeros_like(Z[:, :1, :])
        Zp = tf.concat([Z, tf.repeat(pad, extra, axis=1)], axis=1)

        wave = tf.signal.inverse_stft(
            Zp,
            frame_length=self.FL,
            frame_step=self.FS,
            window_fn=tf.signal.inverse_stft_window_fn(self.FS, forward_window_fn=tf.signal.hann_window),
        )
        wave = wave[:, :self.Lout]
        return wave[..., tf.newaxis]

    def compute_output_shape(self, s):
        return (s[0], self.Lout, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(frame_length=self.FL, frame_step=self.FS, target_len=self.Lout)
        return cfg


# ==================================================================
#  Main model builder
# ==================================================================

def demucs_v4_fixed(input_shape=(441_000, 1), num_instruments=13):
    """
    Build the Hybrid Demucs V4 model for music source separation.
    
    Args:
        input_shape: Audio input shape (samples, channels). Default: 10 seconds @ 44.1kHz mono.
        num_instruments: Number of output stems. Default: 13.
    
    Returns:
        Keras Model with dict outputs keyed 'instrument_1' through 'instrument_13'.
    """

    inputs = Input(shape=input_shape, dtype="float32")

    # 1) Constant padding: 441 000 → 441 024 (divisible by 16)
    x = layers.ZeroPadding1D((12, 12))(inputs)

    # 2) Time‑domain encoder
    skips = []
    x = conv1d_block(x, 32, kernel_size=8, strides=1, use_gelu=True)
    encoder_filters = [64, 128, 256, 512]
    for i, f in enumerate(encoder_filters):
        skips.append(x)
        x = conv1d_block(x, f, kernel_size=4, strides=2, use_gelu=True)
        x = LocalSelfAttention(heads=8, key_dim=f // 8, window_size=1024)(x)

    # 3) Frequency‑domain branch (magnitude only)
    stft = STFT(frame_length=4096, frame_step=1024)(inputs)  # (B, 428, 2049, 2)
    mag = stft[..., 0]
    pha = stft[..., 1]

    f = ExpandDims(axis=-1, name="mag_expand")(mag)  # (B, 428, 2049, 1)
    f = layers.Conv2D(128, 3, padding="same")(f)
    f = layers.BatchNormalization(synchronized=True, dtype="float32")(f)
    f = layers.ReLU()(f)
    f = layers.Conv2D(256, 3, padding="same")(f)
    f = layers.BatchNormalization(synchronized=True, dtype="float32")(f)
    f = layers.ReLU()(f)

    freq_features = ReduceMean(axis=2, name="freq_spatial_reduce")(f)  # (B, 428, 256)

    # 4) Cross‑domain fusion
    time_proj = layers.Dense(512)(x)
    freq_proj = layers.Dense(512)(freq_features)

    t_norm = layers.LayerNormalization()(time_proj)
    f_norm = layers.LayerNormalization()(freq_proj)

    t2f = layers.MultiHeadAttention(num_heads=8, key_dim=64)(t_norm, f_norm)
    time_enh = layers.Add()([time_proj, t2f])

    f2t = layers.MultiHeadAttention(num_heads=8, key_dim=64)(f_norm, t_norm)
    freq_enh = layers.Add()([freq_proj, f2t])

    time_features = layers.Dense(512)(time_enh)
    freq_features = layers.Dense(512)(freq_enh)

    # 5) Bottleneck local transformers (time‑domain)
    for _ in range(4):
        time_features = LocalSelfAttention(heads=8, key_dim=64, window_size=512)(time_features)

    # 6) Decoder
    decoder_filters = [256, 128, 64, 32]
    for i, (filt, skip) in enumerate(zip(decoder_filters, reversed(skips))):
        time_features = layers.Conv1DTranspose(filt, 4, strides=2, padding="same", name=f"up_{i}")(time_features)
        time_features = layers.BatchNormalization(synchronized=True, dtype="float32")(time_features)
        time_features = layers.ReLU()(time_features)

        if skip.shape[-1] != filt:
            skip = layers.Conv1D(filt, 1, padding="same")(skip)
        time_features = layers.Concatenate()([time_features, skip])

        time_features = conv1d_block(time_features, filt, 8)
        time_features = light_conv_block(time_features, filt, 3)

    # Crop back to exactly 441 000 samples
    time_features = conv1d_block(time_features, 32, 8)
    time_features = layers.Cropping1D((12, 12))(time_features)

    # 7) Reconstruct frequency branch → time domain via iSTFT
    mag_hat = layers.Dense(2049)(freq_features)
    mag_hat = ExpandDims(axis=-1, name="mag_hat_expand")(mag_hat)
    pha_e = ExpandDims(axis=-1, name="phase_expand")(pha)

    stft_rec = layers.Concatenate(axis=-1)([mag_hat, pha_e])
    wav_from_freq = InverseSTFT(frame_length=4096, frame_step=1024, target_len=441_000)(stft_rec)

    # 8) Merge the two estimates + final heads
    merged = layers.Concatenate()([time_features, wav_from_freq])
    merged = conv1d_block(merged, 32, 8)

    outputs = {}
    for i in range(num_instruments):
        y = conv1d_block(merged, 16, 3)
        y = layers.Conv1D(1, 1, activation="tanh", padding="same", dtype="float32")(y)
        outputs[f"instrument_{i+1}"] = y

    return Model(inputs, outputs, name="demucs_v4_fixed")


# ==================================================================
#  Loss Function
# ==================================================================

def custom_loss(y_true, y_pred):
    """Mean Squared Error loss for waveform reconstruction."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


# ==================================================================
#  Quick Test
# ==================================================================

if __name__ == "__main__":
    print("Building Hybrid Demucs V4 model...")
    model = demucs_v4_fixed()
    model.summary(line_length=120)
    print("Model built successfully!")