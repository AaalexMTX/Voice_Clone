# encoder/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

#*
# 该模型接收一个 16kHz 的语音波形张量（shape: [B, T]），自动提取梅尔特征 → LSTM 编码 → 全连接 → 单位归一化输出说话人向量（比如 256 维）。
# *#
class SpeakerEncoder(nn.Module):
    def __init__(self, embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=40
        )
        self.instance_norm = nn.InstanceNorm1d(40)

        self.lstm = nn.LSTM(input_size=40, hidden_size=768, num_layers=3, batch_first=True)

        self.linear = nn.Linear(768, embedding_size)

    def forward(self, wav):
        """
        wav: [B, T] 原始波形，采样率已统一为16kHz
        return: [B, embedding_size] speaker embedding
        """
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)

        # 提取梅尔频谱
        mel = self.mel_extractor(wav)  # [B, n_mels, T']
        mel = self.instance_norm(mel)
        mel = mel.transpose(1, 2)  # [B, T', n_mels]

        # LSTM编码
        outputs, _ = self.lstm(mel)  # [B, T', 768]
        embed = outputs[:, -1, :]    # 取最后一个时间步的输出

        embed = self.linear(embed)   # [B, embedding_size]
        embed = F.normalize(embed, dim=1)  # 单位归一化

        return embed
