import os
import subprocess
import wave
import json
import requests
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)

# Input: download link (auto-downloads a .wav file)
INPUT_AUDIO_URL = "https://www.sci.gov.in/?_siwp_play&id=i0ltxn6n3wzt7gzkqaleq2xanahwdbgdu5ix0mn8"
DOWNLOADED_AUDIO = "downloaded.wav"
CONVERTED_AUDIO = "converted.wav"
MODEL_PATH = r"D:\BAPS\CLAW\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"

# Step 1: Download WAV file from URL
def download_audio(url: str, save_path: str) -> bool:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200 and response.content:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

# Step 2: Convert to mono, 16kHz PCM WAV
def convert_audio(input_file, output_file):
    ffmpeg_path = r"D:\BAPS\CLAW\ffmpeg-2025-05-01-git-707c04fe06-full_build\ffmpeg-2025-05-01-git-707c04fe06-full_build\bin\ffmpeg.exe"
    command = [
        ffmpeg_path,
        "-y", "-i", input_file,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Step 3: Transcribe and evaluate
def transcribe_arithmetic(wav_path, model_path) -> int | None:
    try:
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            return None

        model = Model(model_path)
        grammar = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "plus", "minus"
        ]

        rec = KaldiRecognizer(model, wf.getframerate(), json.dumps(grammar))
        results = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                results.append(result.get("text", ""))
        final_result = json.loads(rec.FinalResult())
        results.append(final_result.get("text", ""))

        raw_text = " ".join(results).strip()

        word_to_digit = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
        }
        operations = {"plus": "+", "minus": "-"}

        tokens = raw_text.split()
        if len(tokens) != 3:
            return None

        left = word_to_digit.get(tokens[0])
        op = operations.get(tokens[1])
        right = word_to_digit.get(tokens[2])
        if not left or not op or not right:
            return None

        expr = f"{left}{op}{right}"
        return int(eval(expr))
    except:
        return None

# Main run block
if __name__ == "__main__":
    if download_audio(INPUT_AUDIO_URL, DOWNLOADED_AUDIO):
        convert_audio(DOWNLOADED_AUDIO, CONVERTED_AUDIO)
        result = transcribe_arithmetic(CONVERTED_AUDIO, MODEL_PATH)
    else:
        result = None

    print(result)  # Final output: int or None

