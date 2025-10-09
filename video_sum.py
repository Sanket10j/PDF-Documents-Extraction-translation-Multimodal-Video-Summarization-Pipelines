import os
import subprocess
import torch
import whisper
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configuration
VIDEO_PATH = "video.mp4"
AUDIO_PATH = "audio.wav"
FRAME_DIR = "frames"
FRAME_INTERVAL = 10 
WORDS_PER_SEGMENT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


whisper_model = whisper.load_model("tiny", device=DEVICE)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def download_video(url):
    subprocess.run(["yt-dlp", "-f", "mp4", "-o", VIDEO_PATH, url], check=True)


def extract_audio():
    subprocess.run(["ffmpeg", "-y", "-i", VIDEO_PATH, "-ar", "16000", "-ac", "1", AUDIO_PATH], check=True)


def transcribe_audio():
    result = whisper_model.transcribe(AUDIO_PATH)
    return result["text"]


def extract_frames(interval=FRAME_INTERVAL):
    os.makedirs(FRAME_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_path = os.path.join(FRAME_DIR, f"frame_{idx}.jpg")
            Image.fromarray(frame_rgb).save(image_path)
            frames.append(image_path)
        idx += 1
    cap.release()
    return frames


def segment_text(text, max_words=WORDS_PER_SEGMENT):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def summarize_segment(text_chunk, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text_chunk], images=image, return_tensors="pt", padding=True).to(DEVICE)
    outputs = clip_model(**inputs)
    similarity = torch.nn.functional.cosine_similarity(outputs.text_embeds, outputs.image_embeds)
    return f"Summary relevance score: {similarity.item():.2f}"


def summarize_youtube_video(url):
    print(" Downloading vid")
    download_video(url)

    print("🔊 Extracting aud")
    extract_audio()

    print("📝 Transcribing audi")
    transcript = transcribe_audio()

    print("🖼️ Extracting frame")
    frames = extract_frames()

    print("✂️ Segmenting transcript")
    segments = segment_text(transcript)

    print("🧠 Generating summaries")
    results = []
    for i, (text_chunk, image_path) in enumerate(zip(segments, frames)):
        print(f"\n[Segment {i}]")
        summary = summarize_segment(text_chunk, image_path)
        print(f"→ {summary}")
        results.append({
            "segment": i,
            "image": image_path,
            "summary": summary,
            "text": text_chunk
        })

    return results

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=p_o4aY7xkXg"
    summarize_youtube_video(youtube_url)


