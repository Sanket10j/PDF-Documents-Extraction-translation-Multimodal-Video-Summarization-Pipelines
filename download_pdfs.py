import os
from pathlib import Path
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

# -------------------- CONFIGURATION --------------------

GCS_BUCKET_NAME = "ecourts_hccourts_pdf_extraction_bucket"
PDF_PREFIX = "bombay_hc_court_test/"
LOCAL_OUTPUT_DIR = Path("local_bombay_pdfs")  # Local folder to save PDFs

# ------------------------------------------------------

def download_all_pdfs():
    # Create local output directory
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    print(f"Listing blobs in '{PDF_PREFIX}' ...")
    try:
        blobs = list(bucket.list_blobs(prefix=PDF_PREFIX))
    except GoogleAPIError as e:
        print(f"Error listing blobs: {e}")
        return

    print(f"Found {len(blobs)} blobs under '{PDF_PREFIX}'.")

    pdf_blobs = [blob for blob in blobs if blob.name.lower().endswith(".pdf")]
    print(f"Found {len(pdf_blobs)} PDF files.")

    for i, blob in enumerate(pdf_blobs, 1):
        pdf_name = Path(blob.name).name
        local_path = LOCAL_OUTPUT_DIR / pdf_name

        print(f"[{i}/{len(pdf_blobs)}] Downloading '{pdf_name}' ...", end=" ")
        try:
            blob.download_to_filename(str(local_path))
            print("✅ Done")
        except Exception as e:
            print(f"⚠ Failed: {e}")

    print("✅ All PDFs downloaded to:", LOCAL_OUTPUT_DIR.resolve())

if __name__ == "__main__":
    print("🚀 Starting download of Bombay HC Court PDFs ...")
    download_all_pdfs()
    print("🎉 Finished.")
