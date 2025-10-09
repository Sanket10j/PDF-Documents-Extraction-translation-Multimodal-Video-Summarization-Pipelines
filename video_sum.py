# import os
# import subprocess
# import whisper
# import torch
# import torchaudio
# import cv2
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForCausalLM
# from moviepy.editor import VideoFileClip

# # --- Step 1: Download YouTube video ---
# def download_video(url, output_path='video.mp4'):
#     subprocess.run(['yt-dlp', '-f', 'mp4', '-o', output_path, url])

# # --- Step 2: Extract audio and frames ---
# def extract_audio(video_path, audio_path='audio.wav'):
#     subprocess.run(['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1', audio_path])

# def extract_frames(video_path, interval_sec=10, frame_dir='frames'):
#     os.makedirs(frame_dir, exist_ok=True)
#     clip = VideoFileClip(video_path)
#     for t in range(0, int(clip.duration), interval_sec):
#         frame = clip.get_frame(t)
#         frame_img = Image.fromarray(frame)
#         frame_img.save(f'{frame_dir}/frame_{t}.jpg')

# # --- Step 3: Transcribe using Whisper ---
# def transcribe_audio(audio_path):
#     model = whisper.load_model("base")
#     result = model.transcribe(audio_path)
#     return result["text"]

# # --- Step 4: Segment transcript (fixed length or Whisper chunks) ---
# def segment_text(text, max_words=100):
#     words = text.split()
#     segments = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
#     return segments

# # --- Step 5: Use Cosmos for summarization ---
# def summarize_with_cosmos(segment, image_path=None):
#     processor = AutoProcessor.from_pretrained("microsoft/cosmos-2")
#     model = AutoModelForCausalLM.from_pretrained("microsoft/cosmos-2")

#     inputs = processor(text=segment, images=Image.open(image_path) if image_path else None, return_tensors="pt")
#     generate_ids = model.generate(**inputs, max_length=80)
#     summary = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
#     return summary

# # --- Main pipeline ---
# def process_video(youtube_url):
#     download_video(youtube_url)
#     extract_audio('video.mp4')
#     extract_frames('video.mp4', interval_sec=10)

#     transcript = transcribe_audio('audio.wav')
#     segments = segment_text(transcript, max_words=100)

#     results = []
#     frame_files = sorted(os.listdir('frames'))
#     for i, segment in enumerate(segments):
#         image_path = f'frames/{frame_files[i % len(frame_files)]}' if frame_files else None
#         summary = summarize_with_cosmos(segment, image_path)
#         results.append({
#             "segment": i,
#             "headline": summary.split('.')[0],
#             "summary": summary
#         })

#     return results

# # Example usage
# youtube_url = "https://www.youtube.com/watch?v=your_video_id"
# video_summaries = process_video(youtube_url)

# for entry in video_summaries:
#     print(f"[Segment {entry['segment']}] {entry['headline']}")
#     print(f"→ {entry['summary']}\n")



####################################################################

# import os
# import subprocess
# import torch
# import whisper
# import av
# import numpy as np
# import cv2
# from huggingface_hub import hf_hub_download
# from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
# from PIL import Image

# # Configuration
# VIDEO_PATH = "video.mp4"
# AUDIO_PATH = "audio.wav"
# FRAME_DIR = "frames"
# FRAME_PER_SEGMENT = 8
# WORDS_PER_SEGMENT = 120
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

# # Load the model and processor
# model = LlavaNextVideoForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
# ).to(DEVICE)

# processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)

# # Step 1: Download YouTube video
# def download_youtube_video(url):
#     subprocess.run(['yt-dlp', '-f', 'mp4', '-o', VIDEO_PATH, url])

# # Step 2: Extract audio
# def extract_audio():
#     subprocess.run(['ffmpeg', '-y', '-i', VIDEO_PATH, '-ar', '16000', '-ac', '1', AUDIO_PATH])

# # Step 3: Transcribe using Whisper
# def transcribe_audio():
#     model = whisper.load_model("base")
#     result = model.transcribe(AUDIO_PATH)
#     return result["text"]

# # Step 4a: Read frames using PyAV
# def read_video_pyav(container, indices):
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# # Step 4b: Extract N frames evenly spaced
# def extract_frames():
#     os.makedirs(FRAME_DIR, exist_ok=True)
#     container = av.open(VIDEO_PATH)
#     video_stream = container.streams.video[0]

#     # Fallback to OpenCV if frame count is unavailable
#     total_frames = video_stream.frames
#     if total_frames <= 0:
#         cap = cv2.VideoCapture(VIDEO_PATH)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()

#     indices = np.linspace(0, total_frames - 1, FRAME_PER_SEGMENT).astype(int)
#     return read_video_pyav(container, indices)

# # Step 5: Segment text into chunks
# def segment_text(text, max_words=120):
#     words = text.split()
#     return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# # Step 6: Summarize each segment
# def summarize_segment(video_clip, segment_caption):
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": segment_caption},
#                 {"type": "video"},
#             ],
#         },
#     ]
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs_video = processor(text=prompt, videos=video_clip, padding=True, return_tensors="pt").to(model.device)
#     output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
#     return processor.decode(output[0][2:], skip_special_tokens=True)

# # Full pipeline
# def summarize_youtube_video(youtube_url):
#     print("Downloading video...")
#     download_youtube_video(youtube_url)

#     print("Extracting audio...")
#     extract_audio()

#     print("Transcribing audio...")
#     transcript = transcribe_audio()

#     print("Extracting video frames...")
#     frames = extract_frames()

#     print("Segmenting transcript...")
#     segments = segment_text(transcript, max_words=WORDS_PER_SEGMENT)

#     print("Generating summaries using LLaVA-NeXT-Video-7B-hf...")
#     results = []
#     for i, segment in enumerate(segments):
#         print(f"\n[Segment {i}] Summarizing...")
#         video_clip = frames  # Use the same frames for each segment
#         summary = summarize_segment(video_clip, segment)
#         results.append({
#             "segment": i,
#             "headline": summary.split('.')[0],
#             "summary": summary
#         })
#         print(f"→ {summary}\n")

#     return results

# # Run it
# if __name__ == "__main__":
#     youtube_url = "https://www.youtube.com/watch?v=p_o4aY7xkXg"  # Replace with your link
#     summaries = summarize_youtube_video(youtube_url)

#     for summary in summaries:
#         print(f"Segment {summary['segment']} Headline: {summary['headline']}")
#         print(f"Summary: {summary['summary']}")



#############################################################

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
FRAME_INTERVAL = 10  # seconds
WORDS_PER_SEGMENT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper (tiny) and CLIP (ViT-B/32)
whisper_model = whisper.load_model("tiny", device=DEVICE)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Step 1: Download YouTube Video
def download_video(url):
    subprocess.run(["yt-dlp", "-f", "mp4", "-o", VIDEO_PATH, url], check=True)

# Step 2: Extract audio from video
def extract_audio():
    subprocess.run(["ffmpeg", "-y", "-i", VIDEO_PATH, "-ar", "16000", "-ac", "1", AUDIO_PATH], check=True)

# Step 3: Transcribe audio using Whisper
def transcribe_audio():
    result = whisper_model.transcribe(AUDIO_PATH)
    return result["text"]

# Step 4: Extract frames from video
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

# Step 5: Segment transcript into chunks
def segment_text(text, max_words=WORDS_PER_SEGMENT):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Step 6: Summarize each text + image pair
def summarize_segment(text_chunk, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text_chunk], images=image, return_tensors="pt", padding=True).to(DEVICE)
    outputs = clip_model(**inputs)
    similarity = torch.nn.functional.cosine_similarity(outputs.text_embeds, outputs.image_embeds)
    return f"Summary relevance score: {similarity.item():.2f}"

# Step 7: Main pipeline
def summarize_youtube_video(url):
    print("📥 Downloading video...")
    download_video(url)

    print("🔊 Extracting audio...")
    extract_audio()

    print("📝 Transcribing audio...")
    transcript = transcribe_audio()

    print("🖼️ Extracting frames...")
    frames = extract_frames()

    print("✂️ Segmenting transcript...")
    segments = segment_text(transcript)

    print("🧠 Generating summaries...")
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

# Run
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=p_o4aY7xkXg"
    summarize_youtube_video(youtube_url)

