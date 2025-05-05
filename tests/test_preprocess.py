# test_preprocess.py

from encoder.audio_processing import preprocess_wav, save_wav
import os

in_path = "../data/raw/dingzhen_8.wav"
out_path = "../data/processed/clean/dingzhen_8.wav"

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# 预处理并保存
print(f"预处理音频: {in_path}")
clean_wav = preprocess_wav(in_path)
save_wav(clean_wav, out_path)
print(f"保存到: {out_path}")
