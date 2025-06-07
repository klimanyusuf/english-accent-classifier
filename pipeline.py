"""
Accent Classifier Pipeline
Description: Extracts audio from public video URLs, transcribes English speech,
and classifies English accents using Whisper embeddings.
"""

import os
import subprocess
import whisper
import yt_dlp
import numpy as np
import pickle
import torch

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load pre-trained classifier (mock)
with open("classifier.pkl", "rb") as f:
    accent_model = pickle.load(f)

def download_video_from_url(url):
    video_path = "temp_video.mp4"
    ydl_opts = {
        'outtmpl': video_path,
        'format': 'mp4/bestaudio/best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return video_path

def extract_audio_track(video_file_path):
    audio_output_path = "temp_audio.wav"
    subprocess.run([
        "ffmpeg", "-i", video_file_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_output_path, "-y"
    ], check=True)
    return audio_output_path

def transcribe_audio_file(audio_path):
    transcription_result = whisper_model.transcribe(audio_path, language="en")
    return transcription_result['text'], transcription_result['language']

def get_whisper_embedding(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    
    with torch.no_grad():
        encoder_output = whisper_model.encoder(mel.unsqueeze(0))
    
    embedding_vector = encoder_output.mean(dim=1).cpu().numpy().flatten()
    return embedding_vector

def process_video_url(video_url):
    video_path = download_video_from_url(video_url)
    audio_path = extract_audio_track(video_path)
    
    transcription_text, detected_language = transcribe_audio_file(audio_path)
    if detected_language != 'en':
        return "Non-English", 0.0, f"Detected language: {detected_language}. Accent classification skipped."
    
    embedding = get_whisper_embedding(audio_path).reshape(1, -1)
    probabilities = accent_model.predict_proba(embedding)[0]
    labels = accent_model.classes_
    top_index = np.argmax(probabilities)
    
    accent_label = labels[top_index]
    confidence_score = probabilities[top_index] * 100.0
    
    # Clean up
    os.remove(video_path)
    os.remove(audio_path)
    
    result_summary = (
        f"English speech detected.\n"
        f"Predicted accent: **{accent_label}**\n"
        f"Confidence score: **{confidence_score:.2f}%**\n"
    )
    
    return accent_label, confidence_score, result_summary

