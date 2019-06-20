import numpy as np
from typing import Type
import librosa
from pydub import AudioSegment
import random


def int_samples_to_float(y: np.ndarray, dtype: Type):

    assert isinstance(y, np.ndarray)
    assert issubclass(y.dtype.type, np.integer)
    assert issubclass(dtype, np.floating)

    y = y.astype(dtype) / np.iinfo(y.dtype).max

    return y


def float_samples_to_int(y: np.ndarray, dtype: Type):

    assert isinstance(y, np.ndarray)
    assert issubclass(y.dtype.type, np.floating)
    assert issubclass(dtype, np.integer)

    return (y * np.iinfo(dtype).max).astype(dtype)


def log_mel_energy(inputs: np.ndarray,
                   sr,
                   n_fft=400,
                   stride=160,
                   n_mels=40,
                   freq_min=20,
                   freq_max=8000) -> np.ndarray:
    """ Computes the Log mel filter bank energies of the waveform input"""
    specto = librosa.feature.melspectrogram(
        inputs,
        sr=sr,
        n_fft=n_fft,
        hop_length=stride,
        n_mels=n_mels,
        power=1,  # 1 for energy, 2 for power
        fmin=freq_min,
        fmax=freq_max)
    log_specto = librosa.core.amplitude_to_db(specto, ref=np.max)

    # R -> Time x Freq

    return log_specto.T


def mfcc(data: np.ndarray,
         sample_rate: int = 16000,
         n_mfcc: int = 40,
         stride: int = 20,
         window_size: int = 40):
    """
    computes the mel-frequency cepstral coefficients of the input data
    data - np.float32 ndarray (n,)
    stride - ms
    window_size - ms
    """

    assert isinstance(data, np.ndarray)
    assert isinstance(n_mfcc, int)
    assert isinstance(sample_rate, int)
    assert isinstance(stride, int)
    assert isinstance(window_size, int)
    assert data.dtype.type == np.float32

    stride = int(sample_rate * stride / 1000)
    window_size = int(sample_rate * window_size / 1000)

    result: np.ndarray = librosa.feature.mfcc(y=data,
                                              sr=sample_rate,
                                              n_mfcc=n_mfcc,
                                              hop_length=stride,
                                              n_fft=window_size).astype(
                                                  np.float32)

    # Features x Time > Time x Features

    return result.T


def background_noise_augment(y_overlay: np.ndarray, dBFS: float,
                             bg_noise: AudioSegment, snr_range) -> np.ndarray:
    """Augment by overlaying with background noise"""

    assert isinstance(y_overlay, np.ndarray)
    assert issubclass(y_overlay.dtype.type, np.floating)

    # Select within range
    snr = random.random()
    snr *= (snr_range[1] - snr_range[0])
    target_noise_dBFS = dBFS - snr

    gain = target_noise_dBFS - bg_noise.dBFS
    bg_noise = bg_noise.apply_gain(gain)
    bg_noise = np.array(bg_noise.get_array_of_samples())

    bg_noise = int_samples_to_float(bg_noise, np.float32)

    return y_overlay + bg_noise


def speed_augment(y_speed: np.ndarray) -> np.ndarray:
    "Apply speed augmentation"

    assert isinstance(y_speed, np.ndarray)
    assert issubclass(y_speed.dtype.type, np.floating)

    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed, speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed = np.zeros_like(y_speed)
    y_speed[0:minlen] = tmp[0:minlen]

    return y_speed


def white_noise_augment(y_noise: np.ndarray) -> np.ndarray:
    """ Apply white noise augmentation to the input data"""
    # dBFS
    assert isinstance(y_noise, np.ndarray)
    assert issubclass(y_noise.dtype.type, np.floating)

    noise_amp = 0.005 * np.random.uniform() * np.amax(y_noise)
    y_noise = y_noise + (noise_amp * np.random.normal(size=y_noise.shape[0]))

    return y_noise.astype(np.float32)


def pitch_augment(y_pitch: np.ndarray,
                  sample_rate: int,
                  bins_per_octave: int = 24,
                  pitch_pm: int = 4) -> np.ndarray:

    assert isinstance(y_pitch, np.ndarray)
    assert issubclass(y_pitch.dtype.type, np.floating)

    pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
    y_pitch = librosa.effects.pitch_shift(y_pitch,
                                          sample_rate,
                                          n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)

    return y_pitch


def value_augment(y_aug: np.ndarray, low=0.5, high=1.1) -> np.ndarray:
    """ Randomly distort the audio input by multiplying with random coefficients """

    assert isinstance(y_aug, np.ndarray)
    assert issubclass(y_aug.dtype.type, np.floating)

    dyn_change = np.random.uniform(low=low, high=high)
    y_aug = y_aug * dyn_change

    return y_aug


def random_shift_augment(y_shift: AudioSegment) -> np.ndarray:

    assert isinstance(y_shift, np.ndarray)
    assert issubclass(y_shift.dtype.type, np.floating)

    timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5
                               )  # up to 20% of length
    start = int(y_shift.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y_shift, (start, 0),
                         mode="constant")[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift, (0, -start),
                         mode="constant")[0:y_shift.shape[0]]

    return y_shift


def hpss_augment(y_hpss: np.ndarray, dBFS: float) -> np.ndarray:

    assert isinstance(y_hpss, np.ndarray)
    assert issubclass(y_hpss.dtype.type, np.floating)

    y_hpss = librosa.effects.hpss(y_hpss)

    return y_hpss[1]


def pitch_speed_augment(y_pitch_speed: np.ndarray) -> np.ndarray:

    assert isinstance(y_pitch_speed, np.ndarray)
    assert issubclass(y_pitch_speed.dtype.type, np.floating)

    length_change = np.random.uniform(low=0.5, high=1.5)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac),
                    np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed = np.zeros_like(y_pitch_speed)
    y_pitch_speed[0:minlen] = tmp[0:minlen]

    return y_pitch_speed
