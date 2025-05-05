# preprocess/generate_metadata.py
import os
import pandas as pd


def generate_metadata(csv_path, raw_dir, mel_dir, embedding_dir):
    metadata = []

    # 遍历音频文件，生成 metadata
    for wav_file in os.listdir(raw_dir):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(raw_dir, wav_file)
            mel_path = os.path.join(mel_dir, wav_file.replace(".wav", ".npy"))
            embedding_path = os.path.join(embedding_dir, wav_file.replace(".wav", ".npy"))

            # 假设文本已经处理过
            transcript = "这里是音频的转录文本"

            metadata.append([wav_file, transcript, mel_path, embedding_path])

    # 保存 metadata.csv
    df = pd.DataFrame(metadata, columns=["wav_filename", "transcript", "mel_path", "speaker_embedding_path"])
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")


if __name__ == "__main__":
    raw_dir = "data/raw/"
    mel_dir = "data/processed/mel/"
    embedding_dir = "data/processed/embedding/"
    csv_path = "data/metadata.csv"

    generate_metadata(csv_path, raw_dir, mel_dir, embedding_dir)
