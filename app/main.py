from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os

from app.do_client import call_do_gpt4o, extract_assistant_content

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process", response_class=HTMLResponse)
async def process_file(request: Request, file: UploadFile = File(...)):
    # Simpan file sementara
    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Baca file binary
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    # Prompt ke API GPT-4o
    system_prompt = (
        "You are an OCR assistant. Extract and clean text from the uploaded image."
    )
    response_json = call_do_gpt4o(
        prompt="Extract text from this image.",
        system=system_prompt,
        file_bytes=image_bytes,
        file_type=file.content_type,
    )

    cleaned_text = extract_assistant_content(response_json)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": file.filename,
            "cleaned": cleaned_text,
        },
    )
