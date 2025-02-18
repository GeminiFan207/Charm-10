import numpy as np
import librosa
import librosa.display
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
from transformers import pipeline
import scipy.signal
from speechbrain.pretrained import EncoderClassifier

# Load Audio File
def load_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    return y, sr

# Noise Reduction using Spectral Subtraction
def reduce_noise(y, sr):
    reduced_noise = scipy.signal.medfilt(y, kernel_size=3)
    return reduced_noise

# Convert Speech to Text (ASR)
def speech_to_text(file_path):
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    transcript = asr_model(file_path)
    return transcript["text"]

# Emotion Detection from Voice
def detect_emotion(file_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2", 
                                                savedir="pretrained_models/emotion-recognition")
    signal, sr = torchaudio.load(file_path)
    emotion = classifier.classify_batch(signal)
    return emotion[3]  # Returns predicted emotion

# Sound Classification (e.g., Speech, Music, Noise)
def classify_sound(file_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/sound-classification", 
                                                savedir="pretrained_models/sound-classification")
    signal, sr = torchaudio.load(file_path)
    category = classifier.classify_batch(signal)
    return category[3]  # Returns predicted sound category

# Save Processed Audio
def save_audio(y, sr, output_path):
    librosa.output.write_wav(output_path, y, sr)

# Audio Feature Extraction for AI Model Input
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return {"mfccs": mfccs, "chroma": chroma, "mel_spec": mel_spec}