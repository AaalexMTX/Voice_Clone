# encoder/extract_embedding.py

import os
import numpy as np
from encoder.voice_encoder import VoiceEncoder
from encoder.audio_processing import preprocess_wav

# 给定一条 .wav 音频路径，提取它的 embedding（说话人特征向量），保存成 .npy 文件。
def extract_speaker_embedding(wav_path, embedding_path, encoder=None):
    """
    对单个音频提取 speaker embedding 并保存
    """
    # 如果没传 encoder，就新建一个。
    # 在批量处理时，这样可以避免每次都重新初始化模型，节省开销。
    if encoder is None:
        encoder = VoiceEncoder()

    # 读取 .wav 文件，转换成 PyTorch tensor。
    # 会做一些标准处理：去噪、重采样到 16kHz、归一化。
    wav = preprocess_wav(wav_path)

    # 通过 VoiceEncoder 模型，把语音提取成 256维向量。
    embedding = encoder.embed_utterance(wav)

    # 自动创建输出文件夹（如果不存在
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    # 用 NumPy 保存为 .npy 格式文件，里面是一个浮点数组
    np.save(embedding_path, embedding)
    print(f"✅ Saved: {embedding_path}")

# 批处理所有的 .wav 文件，逐个调用 extract_speaker_embedding。
def extract_all_embeddings(raw_dir, embedding_dir):
    """
    批量提取所有 raw_dir/*.wav 的说话人特征 -> embedding_dir/*.npy
    """
    # 同上，不过换成了批处理wav文件
    encoder = VoiceEncoder()
    os.makedirs(embedding_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(raw_dir) if f.endswith(".wav")]

    # 获取 raw_dir/ 目录下所有 .wav 文件列表
    for wav_file in wav_files:
        wav_path = os.path.join(raw_dir, wav_file)
        embedding_file = wav_file.replace(".wav", ".npy")
        embedding_path = os.path.join(embedding_dir, embedding_file)

        extract_speaker_embedding(wav_path, embedding_path, encoder)


if __name__ == "__main__":
    raw_dir = "data/raw/"
    embedding_dir = "data/processed/embeddings/"

    extract_all_embeddings(raw_dir, embedding_dir)
