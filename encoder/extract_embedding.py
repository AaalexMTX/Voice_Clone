# encoder/extract_embedding.py
import os
import numpy as np
from encoder import VoiceEncoder
from encoder.audio_processing import preprocess_wav


def extract_speaker_embedding(wav_path, embedding_path):
    # 加载说话人编码器
    encoder = VoiceEncoder()

    # 处理音频
    wav, sr = preprocess_wav(wav_path)

    # 提取 speaker embedding
    embedding = encoder.embed_utterance(wav)

    # 保存嵌入
    np.save(embedding_path, embedding)
    print(f"Saved embedding for {wav_path} to {embedding_path}")


if __name__ == "__main__":
    raw_dir = "data/raw/"
    embedding_dir = "data/processed/embedding/"

    # 遍历原始音频文件，生成 speaker embeddings
    for wav_file in os.listdir(raw_dir):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(raw_dir, wav_file)
            embedding_file = wav_file.replace(".wav", ".npy")
            embedding_path = os.path.join(embedding_dir, embedding_file)

            extract_speaker_embedding(wav_path, embedding_path)
