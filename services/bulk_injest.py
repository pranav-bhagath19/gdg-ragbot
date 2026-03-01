import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PDF_FOLDER = os.getenv("PDF_FOLDER", "")
API_URL = "http://localhost:8000"

def upload_pdf(filepath):
    filename = os.path.basename(filepath)
    print(f"\n Uploading: {filename}")
    with open(filepath, "rb") as f:
        response = requests.post(
            f"{API_URL}/ingest",
            files={"file": (filename, f, "application/pdf")}
        )
    if response.status_code != 200:
        print(f"  Failed: {response.text}")
        return None
    job_id = response.json()["job_id"]
    print(f"  Queued. Job ID: {job_id}")
    return job_id

def wait_for_completion(job_id, filename, timeout=120):
    print(f"  Processing {filename}...")
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(f"{API_URL}/ingest/status/{job_id}")
        data = response.json()
        status = data.get("status")
        if status == "completed":
            print(f"  Done! Chunks added: {data.get('chunks_added', 0)}")
            return True
        elif status == "failed":
            print(f"  Failed: {data.get('message', 'unknown error')}")
            return False
        time.sleep(3)
    print(f"  Timeout after {timeout}s")
    return False

def main():
    if not PDF_FOLDER:
        print("ERROR: PDF_FOLDER not set in .env file")
        print("Open .env and add: PDF_FOLDER=C:\\path\\to\\your\\pdfs")
        return

    folder = Path(PDF_FOLDER)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No PDFs found in: {folder}")
        return

    print(f"Found {len(pdfs)} PDFs in: {folder}")
    print("=" * 50)

    success = 0
    failed = 0
    failed_files = []

    for i, pdf_path in enumerate(pdfs, 1):
        filename = pdf_path.name
        print(f"\n[{i}/{len(pdfs)}] {filename}")
        job_id = upload_pdf(str(pdf_path))
        if job_id is None:
            failed += 1
            failed_files.append(filename)
            continue
        ok = wait_for_completion(job_id, filename)
        if ok:
            success += 1
        else:
            failed += 1
            failed_files.append(filename)
        time.sleep(1)

    print("\n" + "=" * 50)
    print(f"Successfully ingested: {success} PDFs")
    print(f"Failed: {failed} PDFs")
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"   - {f}")
    if success > 0:
        print(f"\nReady to chat at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()