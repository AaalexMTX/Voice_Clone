# run_extract_mel.py

from preprocess.mel_extract import extract_all_mel

if __name__ == "__main__":
    clean_dir = "../data/processed/clean"
    mel_dir = "../data/processed/mel"
    extract_all_mel(clean_dir, mel_dir)
