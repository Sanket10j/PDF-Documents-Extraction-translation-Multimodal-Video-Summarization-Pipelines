# # from vosk import Model, KaldiRecognizer
# # import wave
# # import json

# # # Path to your .wav file and the model directory
# # AUDIO_FILE = r"D:\Desktop\Wav\3cuah3.wav"
# # MODEL_PATH = r"C:\Users\BAPS\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"

# # # Load VOSK model
# # model = Model(MODEL_PATH)

# # # Open audio file
# # wf = wave.open(AUDIO_FILE, "rb")

# # # Check format (must be mono PCM, 16-bit, 16kHz)
# # if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
# #     raise ValueError("Audio file must be WAV format: mono, 16-bit, 16000Hz")

# # # Initialize recognizer
# # rec = KaldiRecognizer(model, wf.getframerate())

# # # Read and process audio
# # results = []
# # while True:
# #     data = wf.readframes(4000)
# #     if len(data) == 0:
# #         break
# #     if rec.AcceptWaveform(data):
# #         results.append(json.loads(rec.Result())['text'])

# # # Final result
# # results.append(json.loads(rec.FinalResult())['text'])

# # # Combine and print full transcription
# # full_text = ' '.join(results)
# # print("Transcribed Text:\n", full_text)


# ##########################################################
# import os
# import subprocess
# import wave
# import json
# from vosk import Model, KaldiRecognizer, SetLogLevel

# # Suppress VOSK logs
# SetLogLevel(-1)

# # Input audio (any format)
# INPUT_AUDIO = r"D:\Downloads\securimage_audio-67549c9702d40f49738e4f0ab37d349f.wav"
# # Output audio (VOSK-compatible)
# CONVERTED_AUDIO = "converted.wav"
# # VOSK model path
# MODEL_PATH = r"D:\BAPS\CLAW\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"

# # Step 1: Convert input audio to mono, 16-bit PCM, 16kHz using FFmpeg
# def convert_audio(input_file, output_file):
#     ffmpeg_path = r"D:\BAPS\CLAW\ffmpeg-2025-05-01-git-707c04fe06-full_build\ffmpeg-2025-05-01-git-707c04fe06-full_build\bin\ffmpeg.exe"
#     command = [
#         ffmpeg_path,
#         "-y",
#         "-i", input_file,
#         "-ar", "16000",
#         "-ac", "1",
#         "-sample_fmt", "s16",
#         output_file
#     ]

#     try:
#         subprocess.run(command, check=True)
#         print("✅ Audio converted successfully.")
#     except subprocess.CalledProcessError as e:
#         print("❌ FFmpeg error:", e)
#         raise RuntimeError("FFmpeg failed to convert the audio file.")

# # Step 2: Transcribe audio using limited grammar (letters + digits)
# def transcribe_audio(wav_path, model_path):
#     wf = wave.open(wav_path, "rb")
#     if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
#         raise ValueError("Converted audio must be mono, 16-bit, 16000 Hz.")

#     model = Model(model_path)

#     # Restricted grammar
#     grammar = [
#         "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
#         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
#         "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
#         "ten","plus", "minus"
#     ]

#     rec = KaldiRecognizer(model, wf.getframerate(), json.dumps(grammar))

#     results = []
#     while True:
#         data = wf.readframes(4000)
#         if len(data) == 0:
#             break
#         if rec.AcceptWaveform(data):
#             result = json.loads(rec.Result())
#             results.append(result.get("text", ""))
#     final_result = json.loads(rec.FinalResult())
#     results.append(final_result.get("text", ""))

#     raw_text = " ".join(results)

#     # Map spoken words to digits/symbols
#     word_to_symbol = {
#         "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
#         "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
#         "ten": "10","plus": "+", "minus": "-"
#     }

#     tokens = raw_text.split()
#     normalized = "".join([word_to_symbol.get(token, token) for token in tokens])

#     return normalized

# # Run everything
# if __name__ == "__main__":
#     convert_audio(INPUT_AUDIO, CONVERTED_AUDIO)
#     text = transcribe_audio(CONVERTED_AUDIO, MODEL_PATH)
#     print("\n🔤 Transcribed Alphanumeric Code:\n", text)




###################################
# import os
# import subprocess
# import wave
# import json
# from vosk import Model, KaldiRecognizer, SetLogLevel

# SetLogLevel(-1)

# # Input audio and model path
# INPUT_AUDIO = r"D:\Downloads\securimage_audio-f54cc13bf08da3a2152f767e7d140910.wav"
# CONVERTED_AUDIO = "converted.wav"
# MODEL_PATH = r"D:\BAPS\CLAW\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"

# # FFmpeg audio conversion
# def convert_audio(input_file, output_file):
#     ffmpeg_path = r"D:\BAPS\CLAW\ffmpeg-2025-05-01-git-707c04fe06-full_build\ffmpeg-2025-05-01-git-707c04fe06-full_build\bin\ffmpeg.exe"
#     command = [
#         ffmpeg_path,
#         "-y", "-i", input_file,
#         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
#         output_file
#     ]
#     try:
#         subprocess.run(command, check=True)
#         #print("✅ Audio converted successfully.")
#     except subprocess.CalledProcessError as e:
#         print("❌ FFmpeg error:", e)
#         raise RuntimeError("FFmpeg failed to convert the audio file.")

# # Transcribe arithmetic captcha (e.g., "three plus two")
# def transcribe_arithmetic(wav_path, model_path):
#     wf = wave.open(wav_path, "rb")
#     if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
#         raise ValueError("Audio must be mono, 16-bit, 16000 Hz.")

#     model = Model(model_path)

#     grammar = [
#         "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
#         "plus", "minus"
#     ]

#     rec = KaldiRecognizer(model, wf.getframerate(), json.dumps(grammar))
#     results = []
#     while True:
#         data = wf.readframes(4000)
#         if len(data) == 0:
#             break
#         if rec.AcceptWaveform(data):
#             result = json.loads(rec.Result())
#             results.append(result.get("text", ""))
#     final_result = json.loads(rec.FinalResult())
#     results.append(final_result.get("text", ""))

#     raw_text = " ".join(results).strip()
#     #print("🔤 Raw Transcription:", raw_text)

#     # Mapping
#     word_to_digit = {
#         "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
#         "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
#     }
#     operations = {"plus": "+", "minus": "-"}

#     tokens = raw_text.split()

#     if len(tokens) != 3:
#         raise ValueError("Expected format: [digit] [operation] [digit]")

#     left = word_to_digit.get(tokens[0])
#     op = operations.get(tokens[1])
#     right = word_to_digit.get(tokens[2])

#     if not left or not right or not op:
#         raise ValueError("Invalid words in audio captcha")

#     expr = f"{left}{op}{right}"
#     try:
#         result = eval(expr)
#         return f"{result}"
#     except Exception as e:
#         raise RuntimeError(f"Error evaluating expression: {e}")

# # Run pipeline
# if __name__ == "__main__":
#     convert_audio(INPUT_AUDIO, CONVERTED_AUDIO)
#     result = transcribe_arithmetic(CONVERTED_AUDIO, MODEL_PATH)
#     print(result)


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
