# test_encoder_audio_preprocess.py
import os
import sys
# 获取项目根目录路径
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from encoder.audio_processing import preprocess_wav, save_wav

# 项目目录换成根目录了，所以不是基于test的相对位置
in_path = "./data/raw/dingzhen_8.wav"
out_path = "./data/processed/clean/dingzhen_8.wav"

# in_path = r"E:\Pycharm2024\Code_Item\Voice_Clone_Trans\data\raw\dingzhen_8.wav"
# out_path = r"E:\Pycharm2024\Code_Item\Voice_Clone_Trans\data\processed\clean\dingzhen_8.wav"

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# 预处理并保存
print(f"预处理音频: {in_path}")
clean_wav = preprocess_wav(in_path)
save_wav(clean_wav, out_path)
print(f"保存到: {out_path}")
