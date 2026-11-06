"""Waveform noise, invertible STFT and classifier images."""
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn import functional as F

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 128


def load_audio(path, seconds=5):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // divisor, rate // divisor)
    wave = torch.tensor(np.asarray(wave).copy(), dtype=torch.float32)
    length = int(seconds * SAMPLE_RATE)
    return F.pad(wave[:length], (0, max(0, length - wave.numel())))


def add_noise(wave, snr_db, generator=None):
    power = wave.square().mean()
    if power <= 1e-12:
        raise ValueError("Cannot define SNR for a silent recording")
    noise = torch.randn(wave.shape, generator=generator, device=wave.device)
    noise = noise / noise.square().mean().sqrt().clamp_min(1e-12)
    return wave + noise * (power / (10 ** (snr_db / 10))).sqrt()


def stft(wave):
    return torch.stft(wave, N_FFT, HOP, window=torch.hann_window(N_FFT, device=wave.device), return_complex=True)


def reconstruct(spectrum, length):
    return torch.istft(spectrum, N_FFT, HOP, window=torch.hann_window(N_FFT, device=spectrum.device), length=length)


def image(wave):
    # Fixed dB scale: per-image normalization would hide changes in noise level.
    db = 20 * torch.log10(stft(wave).abs().clamp_min(1e-5))
    normalized = (db.clamp(-80, 40) + 80) / 60 - 1
    return F.interpolate(normalized[None, None], (64, 64), mode="bilinear", align_corners=False)[0]
