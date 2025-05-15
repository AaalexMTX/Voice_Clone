# encoder/voice_encoder.py
# 将预处理过的音频转换成一个说话人特征向量（speaker embedding）

# torch 和 torch.nn：用于搭建神经网络。
# torchaudio：PyTorch 的音频处理库。
# numpy：数值计算库，用于最后把结果转成 numpy 格式。
import torch
import torch.nn as nn
import torchaudio
import numpy as np

# 说话人编码器（Speaker Encoder）的核心部分
# 把一段音频提取成一个 256 维的向量（说话人特征）
class VoiceEncoder(nn.Module):

    # 初始化模型，默认输出维度是 256（说话人特征），LSTM 的隐藏层大小是 768。
    def __init__(self, embedding_dim=256, hidden_size=768):
        # super().__init__() 调用父类构造器
        super().__init__()
        # 一个 三层双向 LSTM，处理 [B, T, 40] 的输入（batch 大小 B，时间帧数 T，特征维度 40）。
        # bidirectional=True 表示每层有两个方向（前后），输出维度变为 hidden_size * 2。
        self.lstm = nn.LSTM(
            input_size=40,       # 40维 log-mel 特征
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        # LSTM 输出是 [B, T, hidden_size*2]。
        # 通过一个全连接层将它降维到 256。
        # 然后接 ReLU 激活函数。
        self.linear = nn.Linear(hidden_size * 2, embedding_dim)
        self.relu = nn.ReLU()

        # 切换到 eval 模式
        # 设置模型为评估模式（evaluation mode），禁用 dropout、batch norm 等训练时特性。
        self.eval()

    def forward(self, mels):
        """
        mels: [B, T, 40]
        """
        # 把 mel 特征送入 LSTM，得到每一帧的编码结果。
        outputs, _ = self.lstm(mels)
        # 对时间维度做平均池化（全局信息），结果 shape 是 [B, hidden_size*2]。
        mean_output = outputs.mean(dim=1)
        # 池化后的结果送入 Linear + ReLU，输出 [B, 256]。
        embed = self.relu(self.linear(mean_output))
        return embed

    def embed_utterance(self, wav_tensor):
        """
        接收 waveform tensor（预处理后），返回 256 维 embedding 向量
        """
        # 提取 log-mel 特征提取 mel 特征 [T, 40]，加一维变成 [1, T, 40]，表示 batch=1。
        mel_spec = self.compute_log_mel(wav_tensor)  # [T, 40]
        mel_spec = mel_spec.unsqueeze(0)  # [1, T, 40]
        # 用模型前向传播提取特征（不求梯度）。
        with torch.no_grad():
            embed = self.forward(mel_spec)
        # 去掉 batch 维度，转成 numpy 返回 [256] 向量。
        return embed.squeeze(0).cpu().numpy()

    def compute_log_mel(self, wav_tensor, sr=16000):
        """
        把 waveform 转换成 log-mel 特征[T, 40]（这是语音模型常用输入）。
        """
        # 用 torchaudio 创建一个 mel-spectrogram 转换器：
        # n_fft=400：窗长 25ms（16kHz 下）。
        # hop_length=160：帧移 10ms。
        # n_mels=40：输出 40 维 mel 特征。
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=400,
            hop_length=160,
            n_mels=40
        )
        # 应用转换器，得到 [40, T] 的 mel 频谱
        mel_spec = mel_spec_transform(wav_tensor)
        # 加 1e-6 防止 log(0)，再取对数。
        # 最后转置成 [T, 40]，供 LSTM 使用。
        log_mel = torch.log(mel_spec + 1e-6).transpose(0, 1)  # [T, 40]
        return log_mel


# 可单元测试，测试模型能不能跑通
if __name__ == "__main__":
    from encoder.audio_processing import preprocess_wav

    wav = preprocess_wav("data/raw/example.wav")
    # 实例化编码器，提取 256 维 embedding，并打印 shape。
    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    print("✅ embedding shape:", embed.shape)  # (256,)

