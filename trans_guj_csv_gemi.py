import pandas as pd
import google.generativeai as genai
import time

# 1. Load your CSV
df = pd.read_csv(r"D:\Downloads\output_000.csv")
df.columns = df.columns.str.strip()

# 2. Set Gemini API Key
genai.configure(api_key="key") 

# 3. Load Gemini model (correct name)
model = genai.GenerativeModel("models/gemini-1.5-flash") 

# 4. Define translation function
def translate_with_gemini(text):
    if pd.isna(text) or not text.strip():
        return text

    prompt = (
        "You will be given a sentence that contains both Gujarati and English text. "
        "Translate **only the Gujarati part** into English. Do **not** change or translate existing English words. "
        "Return the result as a natural English sentence, preserving the original English.\n\n"
        f"Sentence:\n{text}"
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        print(f"Error on: {text}\n{e}")
        return text

# 5. Translate the 'extracted_text' column
df["extracted_text"] = df["extracted_text"].apply(translate_with_gemini)

# 6. Save to new file
df.to_csv("translated_output_gemini.csv", index=False)
print("✅ Gemini translation completed and saved to 'translated_output_gemini.csv'.")
