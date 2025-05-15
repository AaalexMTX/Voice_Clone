# tests/test_encoder_voice_encoder.py

# torch：PyTorch，用来处理张量和深度学习相关操作。
# numpy：处理数组数据结构，如果音频是以 numpy 数组形式返回。
# preprocess_wav：函数用于对原始音频 .wav 文件做预处理（如去噪、重采样），返回干净的语音数据。
import os
import torch
import numpy as np
from encoder.audio_processing import preprocess_wav  # 假设你用的是 resemblyzer 的 encoder/audio.py

#
def test_voice_encoder_embedding():
    print("\nCurrent working dir:", os.getcwd())

    wav_path = "../data/raw/dingzhen_8.wav"
    assert os.path.exists(wav_path), f"Test wav file not found: {wav_path}"

    # 预处理音频
    wav_data = preprocess_wav(wav_path)
    # 检查返回的数据是否是 PyTorch 的 Tensor。
    # 如果是 numpy 数组，就用 torch.from_numpy 转换为 Tensor。
    # 否则直接使用（已经是 tensor）。
    if not isinstance(wav_data, torch.Tensor):
        wav_tensor = torch.from_numpy(wav_data)
    else:
        wav_tensor = wav_data

    assert isinstance(wav_tensor, torch.Tensor), "Preprocessed wav should be a torch.Tensor"

    # 打印处理后的语音张量形状。
    # 一般是类似 [95040] 的一维 tensor，表示有 95040 个采样点。
    print("Preprocessed wav shape:", wav_tensor.shape)