# preprocess/mel_extract.py

import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def wav_to_log_mel(wav, sr=16000, n_mels=80, hop_length=256, win_length=1024):
    """
    输入：wav (np.ndarray)，采样率 sr
    输出：log-mel 频谱 (np.ndarray)，形状 [n_mels, T]
    """
    # 预加重，提升高频
    pre_emphasis = 0.97
    emphasized = np.append(wav[0], wav[1:] - pre_emphasis * wav[:-1])

    # STFT
    spectrogram = librosa.stft(
        emphasized,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length
    )
    spectrogram = np.abs(spectrogram)

    # Mel filterbank
    mel_filter = librosa.filters.mel(sr=sr, n_fft=win_length, n_mels=n_mels)
    mel_spec = np.dot(mel_filter, spectrogram)

    # 转为 log-mel
    log_mel = np.log10(np.maximum(mel_spec, 1e-5))
    return log_mel.astype(np.float32)


def extract_all_mel(clean_dir, mel_dir):
    """
    批量提取 clean/*.wav -> mel/*.npy
    """
    os.makedirs(mel_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]

    for fname in tqdm(wav_files, desc="Extracting mel-spectrograms"):
        wav_path = os.path.join(clean_dir, fname)
        mel_path = os.path.join(mel_dir, fname.replace(".wav", ".npy"))

        # 加载并处理
        wav, sr = librosa.load(wav_path, sr=16000)
        log_mel = wav_to_log_mel(wav, sr)

        # 保存 mel 特征
        np.save(mel_path, log_mel)
