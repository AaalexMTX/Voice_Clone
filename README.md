

# 预设项目结构
### v1
```bash
Voice_Clone_Trans/
│
├── data/                     # 数据存放，原始和处理后都放这里
│   ├── raw/                  # 原始语音数据
│   ├── processed/            # 预处理后的 Mel、embedding 等
│   └── metadata.csv          # 语音数据标注信息（文本、说话人ID等）
│
├── models/                   # 模型结构定义，比如 Transformer 编码器、解码器
│   ├── __init__.py
│   ├── transformer.py
│   ├── vocoder.py            # 如有 HiFi-GAN 或 WaveGlow
│   └── speaker_encoder.py    # 说话人编码模型
│
├── configs/                  # 配置文件，如训练参数、模型参数等
│   ├── train_config.yaml
│   └── model_config.yaml
│
├── encoder/               # 语音预处理脚本
│   ├── audio_processing.py   # librosa、soundfile相关处理
│   ├── extract_mel.py
│   └── extract_speaker_embedding.py
│
├── train/                    # 训练脚本与训练流程
│   ├── trainer.py
│   └── train.py
│
├── inference/                # 推理脚本（语音克隆的 demo）
│   ├── synthesize.py
│   └── demo.wav              # 示例生成语音
│
├── utils/                    # 工具包（日志、可视化、损失函数等）
│   ├── logger.py
│   ├── visualize.py
│   └── losses.py
│
├── results/                  # 实验结果（生成音频、图像等）
│   └── experiment_01/
│       ├── output.wav
│       └── attention_plot.png
│
├── logs/                     # 训练日志、tensorboard文件等
│
├── requirements.txt          # pip依赖列表
├── README.md                 # 项目说明
└── main.py                   # 项目入口


```

### v2
```bash
Voice_Clone_Trans/
├── data/
│   ├── raw/                       # 存放原始语音数据（wav）
│   ├── processed/
│   │   ├── mel/                   # 来自某段音频的 mel 特征
│   │   ├── clean/                 # 对原始音频进行"静音、重采样16Khz、归一化"等操作
│   │   └── embeddings/            # 使用 speaker encoder 模型提取。处理后的音频（clean/）中提取出一个 256 维的嵌入向量。
│   └── metadata.csv               # 每条语音对应说话人ID、文件路径等
├── encoder/                        # 说话人编码器（Speaker Encoder）
│   ├── audio_processing.py        # 基础音频加载、裁剪、采样
│   ├── extract_mel.py             # 从音频生成 mel
├── preprocess/                        # 说话人编码器（Speaker Encoder）
├── train/                    # 训练逻辑
├── config/                   # 配置文件
├── inference/                # 推理脚本（synthesize.py等）
├── utils/                    # 工具代码
├── results/                  # 结果输出
├── logs/                     # 日志
├── tests/                     # 测试文件
│
├── go_backend/               # ✅ Go 实现的中间件服务
│   ├── cmd/
│   │   └── main.go           # Go 入口
│   ├── internal/
│   │   ├── handler/          # HTTP/gRPC 接口
│   │   ├── client/           # 与 Python 通信（HTTP 请求）
│   │   └── service/          # 后端逻辑
│   └── go.mod
│
├── web/                      # ✅ 前端界面（React/Vue/HTML）
│
├── requirements.txt
├── README.md
└── main.py                   # Python 项目入口（模型推理接口用）

```