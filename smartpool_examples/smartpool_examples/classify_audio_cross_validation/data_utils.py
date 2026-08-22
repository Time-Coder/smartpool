import os
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from config import (
    ESC50_URLS, FEATURES_DIR, HOP_LENGTH, MEL_WIDTH, N_MELS, NPERSEG, SAMPLE_RATE,
)
from scipy.io import wavfile
from scipy.signal import stft

ESC50_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'esc50')
WAVS_DIR = os.path.join(ESC50_DIR, 'wavs')
PARQUET_FILES = [
    'train-00000-of-00001-cd782ca55710a2e6.parquet',
    'test-00000-of-00001-f167dc83a7b3449c.parquet',
]

_MEL_FILTERBANK = None


def _download_with_resume(url, dest):
    expected = None
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        expected = os.path.getsize(dest)
    if expected is not None and expected > 0:
        return

    tmp = dest + '.part'
    mode = 'ab' if os.path.exists(tmp) else 'wb'
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {'Range': f'bytes={start}-'} if start > 0 else {}

    print(f"[data_utils] downloading {os.path.basename(dest)}...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, mode) as f:
        t0 = time.time()
        total = start
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            speed = total / max(time.time() - t0, 1e-6)
            print(f"\r  {total/1e6:.1f} MB  {speed/1e6:.1f} MB/s", end='', flush=True)
    print()
    os.replace(tmp, dest)


def _ensure_downloaded():
    os.makedirs(ESC50_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    urls = ESC50_URLS[0]
    for fname in PARQUET_FILES:
        dest = os.path.join(ESC50_DIR, fname)
        if not os.path.exists(dest):
            _download_with_resume(urls['train' if 'train' in fname else 'test'], dest)


def _read_parquet_table(fname):
    _ensure_downloaded()
    return pq.read_table(os.path.join(ESC50_DIR, fname))


def _mel_filterbank(n_mels, n_fft, sr):
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = _build_mel_filterbank(n_mels, n_fft, sr)
    return _MEL_FILTERBANK


def _build_mel_filterbank(n_mels, n_fft, sr):
    f_min = 0.0
    f_max = sr / 2.0
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_freqs = np.linspace(f_min, f_max, n_freqs)

    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower, center, upper = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, f in enumerate(bin_freqs):
            if f < lower or f > upper:
                continue
            if f <= center:
                filterbank[i, j] = (f - lower) / (center - lower)
            else:
                filterbank[i, j] = (upper - f) / (upper - center)
    return filterbank


def _log_mel_spectrogram_from_file(wav_path, target_len):
    sr, samples = wavfile.read(wav_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32) / 32768.0

    if sr != SAMPLE_RATE:
        import scipy.signal as ss
        samples = ss.resample_poly(samples, SAMPLE_RATE, sr)

    f, t, Zxx = stft(samples, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NPERSEG - HOP_LENGTH)
    mag = np.abs(Zxx)
    fb = _mel_filterbank(N_MELS, NPERSEG, SAMPLE_RATE)
    mel = fb @ mag
    mel = np.log(mel + 1e-6)

    n_t = mel.shape[1]
    if n_t < target_len:
        pad = np.zeros((N_MELS, target_len - n_t), dtype=mel.dtype)
        mel = np.concatenate([mel, pad], axis=1)
    elif n_t > target_len:
        step = n_t / target_len
        idx = (np.arange(target_len) * step).astype(int)
        idx = np.clip(idx, 0, n_t - 1)
        mel = mel[:, idx]

    mean = mel.mean()
    std = mel.std() + 1e-6
    mel = (mel - mean) / std

    return mel.astype(np.float32)


def extract_wavs_to_disk(meta_rows):
    os.makedirs(WAVS_DIR, exist_ok=True)
    to_extract = [row for row in meta_rows if not os.path.exists(os.path.join(WAVS_DIR, row['filename']))]
    if not to_extract:
        print(f"[data_utils] all {len(meta_rows)} WAVs already extracted")
        return
    print(f"[data_utils] extracting {len(to_extract)} WAVs to disk...")
    for i, row in enumerate(to_extract):
        audio_bytes = row['audio']
        wav_path = os.path.join(WAVS_DIR, row['filename'])
        with open(wav_path, 'wb') as f:
            f.write(audio_bytes)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(to_extract)}")
    print(f"  done ({len(to_extract)} files)")


def preprocess_audio(fname):
    wav_path = os.path.join(WAVS_DIR, fname)
    feat = _log_mel_spectrogram_from_file(wav_path, target_len=MEL_WIDTH)
    out_path = os.path.join(FEATURES_DIR, os.path.splitext(fname)[0] + '.npy')
    np.save(out_path, feat)


def prepare_metadata():
    """Load all ESC-50 rows (train + test parquet) into one metadata list (no audio bytes)."""
    rows = []
    for fname in PARQUET_FILES:
        table = _read_parquet_table(fname)
        for row in table.to_pylist():
            rows.append({
                'filename': row['filename'],
                'fold': int(row['fold']),
                'target': int(row['target']),
                'category': row['category'],
                'audio': row['audio']['bytes'],
            })
    return rows


def load_features(meta_rows):
    """Return (feature_tensor, target_tensor) stacked from saved .npy files."""
    import torch
    feats = []
    targets = []
    for row in meta_rows:
        npy_path = os.path.join(FEATURES_DIR, os.path.splitext(row['filename'])[0] + '.npy')
        feat = np.load(npy_path)
        feats.append(torch.from_numpy(feat).unsqueeze(0).unsqueeze(0))
        targets.append(row['target'])
    data = torch.cat(feats, dim=0)
    data.share_memory_()
    tgt = torch.tensor(targets, dtype=torch.long)
    tgt.share_memory_()
    return data, tgt