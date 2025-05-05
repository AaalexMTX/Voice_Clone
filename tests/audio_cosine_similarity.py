#audio cosine similarity
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine

encoder = VoiceEncoder()
emb1 = encoder.embed_utterance(preprocess_wav("target.wav"))
emb2 = encoder.embed_utterance(preprocess_wav("synthesized.wav"))
similarity = 1 - cosine(emb1, emb2)
