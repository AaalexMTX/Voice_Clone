# preprocess/extract_mel.py
import os
import librosa
import numpy as np
import soundfile as sf
from encoder.audio_processing import preprocess_wav  # 用你之前编写的预处理函数


def extract_mel(wav_path, mel_path):
    # 加载音频
    wav, sr = preprocess_wav(wav_path)

    # 计算 mel-spectrogram
    mel = librosa.feature.melspectrogram(wav, sr=sr, n_mels=80, fmax=8000)
    mel = librosa.power_to_db(mel, ref=np.max)

    # 保存 mel-spectrogram
    np.save(mel_path, mel)


if __name__ == "__main__":
    raw_dir = "data/raw/"
    mel_dir = "data/processed/mel/"

    # 遍历原始音频文件，生成 mel-spectrogram
    for wav_file in os.listdir(raw_dir):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(raw_dir, wav_file)
            mel_file = wav_file.replace(".wav", ".npy")
            mel_path = os.path.join(mel_dir, mel_file)

            extract_mel(wav_path, mel_path)
            print(f"Extracted Mel for {wav_file} to {mel_path}")
