PDF-Documents-Extraction-Translation-Multimodal-Video-Summarization-Pipelines

1. Legal PDF Text Extraction & Summarization Pipeline:

An end-to-end autonomous pipeline for processing and summarizing legal documents written in regional languages.

Key Features:

OCR & Text Extraction: Extracts text from scanned and non-scanned legal PDF documents using open-source Hugging Face OCR models.

Multilingual Translation: Translates extracted text from regional Indian languages into English using different open-source transformer models.

Summarization: Generates concise, structured English summaries of lengthy legal texts using transformer-based summarization models.

Automated Data Handling: Automatically appends extracted, translated, and summarized outputs to CSV files for organized storage, easy retrieval, and downstream analysis.

2. Multimodal Video Summarization Pipeline:

A robust multimodal system designed to generate comprehensive summaries from videos by combining visual and audio information.

Key Features:

Speech Transcription: Utilizes OpenAI Whisper for accurate audio-to-text transcription.

Visual Analysis: Extracts on-screen text and key visual cues from video frames using image captioning and OCR models (e.g., BLIP-2 or Tesseract).

Multimodal Fusion: Integrates textual, auditory, and visual information to produce coherent, topic-structured summaries.

Applications: Ideal for summarizing lectures, legal hearings, documentaries, and other content-rich videos.
