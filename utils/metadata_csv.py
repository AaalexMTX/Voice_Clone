import os
import csv

# 音频文件和文本文件的目录
audio_dir = '../data/raw/'
text_dir = '../data/texts/'  # 假设文本文件与音频文件在同一目录下或有不同的目录

# 输出文件路径
metadata_file = 'metadata.csv'

# 打开 CSV 文件并写入标题行
with open(metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['wav_path', 'text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # 遍历音频文件
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):  # 只处理 .wav 文件
            wav_path = os.path.join(audio_dir, audio_file)

            # 假设每个音频文件都有一个对应的文本文件
            text_file = audio_file.replace('.wav', '.txt')  # 假设文本文件的名字与音频文件一致

            # 检查对应的文本文件是否存在
            text_path = os.path.join(text_dir, text_file)
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()  # 读取文本并去除首尾空白
                writer.writerow({'wav_path': wav_path, 'text': text})
            else:
                print(f"Warning: No text file found for {audio_file}")

print(f"Metadata CSV file '{metadata_file}' has been generated.")
