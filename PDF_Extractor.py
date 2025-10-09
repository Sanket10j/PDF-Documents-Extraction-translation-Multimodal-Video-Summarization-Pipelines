# Dependencies:
# System dependencies
# apt-get -y install poppler-utils
# wget https://github.com/tesseract-ocr/tessdata/raw/main/guj.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/

# # Python libraries
# pip install -q pdf2image pytesseract deep-translator transformers accelerate bitsandbytes
# pip install -q torch torchvision torchaudio pandas pillow opencv-python sentencepiece






import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import re
import textwrap
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)

# ========================== CONFIG ========================== #
PDF_PATH = "/kaggle/input/legalaa/12.pdf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models
TRANSLATION_MODEL = "facebook/nllb-200-1.3B"
SUMMARY_MODEL = "facebook/bart-large-cnn"
EXTRACTION_MODEL = "google/flan-t5-base"

# ========================== OCR PREPROCESSING ========================== #
def preprocess_image_for_tesseract(pil_img):
    img = np.array(pil_img.convert('L'))  # Grayscale
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Denoise
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=450)
    combined_text = []
    for page in pages:
        preprocessed = preprocess_image_for_tesseract(page)
        text = pytesseract.image_to_string(preprocessed, lang="guj+eng")
        combined_text.append(text)
    raw_text = "\n".join(combined_text)
    return re.sub(r'\s+', ' ', raw_text).strip()

# ========================== TRANSLATION ========================== #
def load_translation_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
    device_id = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model, tokenizer=tokenizer, src_lang="guj_Gujr", tgt_lang="eng_Latn", device=device_id)

def translate_text(text, translator, max_chunk_len=512):
    chunks = [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
    translations = []
    for chunk in chunks:
        if len(chunk.strip()) < 5:
            continue
        try:
            out = translator(chunk, max_length=512)[0]["translation_text"]
            translations.append(out)
        except Exception as e:
            translations.append(f"[Translation Error]: {e}")
    return " ".join(translations)

# ========================== SUMMARIZATION ========================== #
def load_summarization_model():
    tokenizer = BartTokenizer.from_pretrained(SUMMARY_MODEL)
    model = BartForConditionalGeneration.from_pretrained(SUMMARY_MODEL).to(DEVICE)
    return tokenizer, model

def chunk_text(text, max_tokens=1024):
    return textwrap.wrap(text, width=max_tokens, break_long_words=False)

def summarize_text(text, tokenizer, model):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        summary_ids = model.generate(inputs["input_ids"], max_length=256, min_length=30, num_beams=4)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)

# ========================== EXTRACTION ========================== #
def load_extraction_model():
    tokenizer = T5Tokenizer.from_pretrained(EXTRACTION_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(EXTRACTION_MODEL).to(DEVICE)
    return tokenizer, model

def extract_fields_t5(text, tokenizer, model):
    prompt = f"""
Extract the following details from the legal judgment:

- Case Number
- Date of Judgment
- Petitioner / Plaintiff
- Respondent / Defendant

Text:
{text}
"""
    inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    fields = {
        "Case Number": "",
        "Date of Judgment": "",
        "Petitioner / Plaintiff": "",
        "Respondent / Defendant": "",
    }
    for line in response.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if key in fields:
                fields[key] = val
    return fields

def extract_fields_regex(text, current_fields):
    if not current_fields["Case Number"]:
        match = re.search(r"(Criminal\s+)?Case\s+No[:.]?\s*(\S+)", text, re.IGNORECASE)
        if match:
            current_fields["Case Number"] = match.group(2)

    if not current_fields["Date of Judgment"]:
        match = re.search(r"Judgment\s+is\s+dated\s+([0-9]{2}/[0-9]{2}/[0-9]{4})", text, re.IGNORECASE)
        if not match:
            match = re.search(r"Date\s+of\s+Judgment[:.]?\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", text, re.IGNORECASE)
        if match:
            current_fields["Date of Judgment"] = match.group(1)

    if not current_fields["Petitioner / Plaintiff"]:
        match = re.search(r"complainant[:,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
        if match:
            current_fields["Petitioner / Plaintiff"] = match.group(1)

    if not current_fields["Respondent / Defendant"]:
        match = re.search(r"accused[:,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
        if match:
            current_fields["Respondent / Defendant"] = match.group(1)

    return current_fields

# ========================== PIPELINE ========================== #
def process_document(full_text, translator, sum_tokenizer, sum_model, ext_tokenizer, ext_model):
    translated_text = translate_text(full_text, translator)
    summary = summarize_text(translated_text, sum_tokenizer, sum_model)
    fields = extract_fields_t5(translated_text, ext_tokenizer, ext_model)
    fields = extract_fields_regex(translated_text, fields)
    fields["Summary of Decision"] = summary
    return fields

def run_pipeline(pdf_path):
    print("🔍 Extracting text from PDF...")
    gujarati_text = extract_text_from_pdf(pdf_path)

    print("🌐 Loading models...")
    translator = load_translation_pipeline()
    sum_tokenizer, sum_model = load_summarization_model()
    ext_tokenizer, ext_model = load_extraction_model()

    print("⚙️  Processing document...")
    data = process_document(gujarati_text, translator, sum_tokenizer, sum_model, ext_tokenizer, ext_model)
    df = pd.DataFrame([data])

    output_csv = "legal_summary_output.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV saved as '{output_csv}'")
    return df

# ========================== EXECUTE ========================== #
if __name__ == "__main__":
    run_pipeline(PDF_PATH)
