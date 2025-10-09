import easyocr
import cv2
import numpy as np
import os
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import moviepy.editor as mp
from moviepy.editor import VideoFileClip
import whisper
import tempfile
import shutil

def extract_frames_from_video(video_path, output_folder, frame_interval=1):
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % int(fps * frame_interval) == 0:
            frame_path = output_folder / f"frame_{saved_frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_frame_count

def extract_text_from_image(image_path):
    
    reader = easyocr.Reader(['en'], gpu=False)  
    
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    results = reader.readtext(thresh)
    
    
    extracted_text = []
    for (bbox, text, prob) in results:
        extracted_text.append({
            'text': text,
            'confidence': prob,
            'bounding_box': bbox
        })
    
    return extracted_text

def process_frames(folder_path):
   
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder {folder_path} does not exist or is not a directory.")
    
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    
    all_results = {}
    
    
    for file_path in sorted(folder.iterdir()): 
        if file_path.suffix.lower() in image_extensions:
            try:
                print(f"Processing frame: {file_path.name}")
                text_data = extract_text_from_image(file_path)
                all_results[file_path.name] = text_data
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
    
    return all_results

def extract_audio_and_transcribe(video_path, temp_dir):
    
    video = VideoFileClip(str(video_path))
    audio_path = temp_dir / "audio.wav"
    video.audio.write_audiofile(str(audio_path))
    video.close()
    
    
    model = whisper.load_model("base")  
    result = model.transcribe(str(audio_path))
    transcript = result["text"]
    
    return transcript

def generate_bart_summary(text, max_length=150, min_length=30):
    if not text.strip():
        return "No text available to summarize."
    
    
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
   
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    
    
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        early_stopping=True
    )
    
   
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_video_summary(frame_summary, transcript_summary):
   
    combined_text = f"Visual Content: {frame_summary}\nAudio Content: {transcript_summary}"
    return generate_bart_summary(combined_text, max_length=200, min_length=50)

def save_results_to_file(frame_results, transcript, summary, output_file):
   
    with open(output_file, 'w', encoding='utf-8') as f:
        
        f.write("Detailed Frame OCR Results\n")
        f.write("=" * 50 + "\n")
        for frame_name, text_data in frame_results.items():
            f.write(f"\nFrame: {frame_name}\n")
            f.write("-" * 50 + "\n")
            if text_data:
                for item in text_data:
                    f.write(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}\n")
            else:
                f.write("No text detected.\n")
        
        
        f.write("\nAudio Transcript\n")
        f.write("=" * 50 + "\n")
        f.write(transcript if transcript.strip() else "No audio transcript available.\n")
        
        
        f.write("\nSummary of Video\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Frames Processed: {summary['total_frames']}\n")
        f.write(f"Frames with Text: {summary['frames_with_text']}\n")
        f.write(f"Frames without Text: {summary['frames_without_text']}\n")
        f.write("\nFrame Text Summary:\n")
        f.write(f"{summary['frame_summary']}\n")
        f.write("\nAudio Transcript Summary:\n")
        f.write(f"{summary['transcript_summary']}\n")
        f.write("\nCombined Video Summary:\n")
        f.write(f"{summary['video_summary']}\n")

def main():
    video_path = 'test2.mp4'  
    output_file = 'video_ocr_transcript_summary.txt'  
    temp_dir = Path(tempfile.mkdtemp())  
    
    try:
        
        frames_folder = temp_dir / "frames"
        print("Extracting frames from video...")
        total_frames = extract_frames_from_video(video_path, frames_folder, frame_interval=1)
        
        
        print("\nProcessing frames for OCR...")
        frame_results = process_frames(frames_folder)
        
        
        print("\nExtracting and transcribing audio...")
        transcript = extract_audio_and_transcribe(video_path, temp_dir)
        
    
       
        all_frame_text = []
        for frame_name, text_data in frame_results.items():
            for item in text_data:
                all_frame_text.append(item['text'])
        frame_text_combined = " ".join(all_frame_text)
        
       
        frame_summary = generate_bart_summary(frame_text_combined)
        
      
        transcript_summary = generate_bart_summary(transcript)
        
        
        video_summary = generate_video_summary(frame_summary, transcript_summary)
        
    
        summary = {
            'total_frames': total_frames,
            'frames_with_text': sum(1 for text_data in frame_results.values() if text_data),
            'frames_without_text': total_frames - sum(1 for text_data in frame_results.values() if text_data),
            'frame_summary': frame_summary,
            'transcript_summary': transcript_summary,
            'video_summary': video_summary
        }
        
       
        save_results_to_file(frame_results, transcript, summary, output_file)
        print(f"\nResults and summary saved to {output_file}")
        
        
        print("\nDetailed Frame OCR Results")
        print("=" * 50)
        for frame_name, text_data in frame_results.items():
            print(f"\nFrame: {frame_name}")
            print("-" * 50)
            if text_data:
                for item in text_data:
                    print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
            else:
                print("No text detected.")
        
        print("\nAudio Transcript")
        print("=" * 50)
        print(transcript if transcript.strip() else "No audio transcript available.")
        
        print("\nSummary of Video")
        print("=" * 50)
        print(f"Total Frames Processed: {summary['total_frames']}")
        print(f"Frames with Text: {summary['frames_with_text']}")
        print(f"Frames without Text: {summary['frames_without_text']}")
        print("\nFrame Text Summary:")
        print(summary['frame_summary'])
        print("\nAudio Transcript Summary:")
        print(summary['transcript_summary'])
        print("\nCombined Video Summary:")
        print(summary['video_summary'])
                
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
       
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
