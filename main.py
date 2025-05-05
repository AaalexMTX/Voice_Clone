from encoder.audio_processing import preprocess_wav, save_wav

# 假设原始语音路径
in_path = "data/raw/spk1_001.wav"
out_path = "data/processed/clean/spk1_001.wav"

# 预处理 + 保存
clean_wav = preprocess_wav(in_path)
save_wav(clean_wav, out_path)
