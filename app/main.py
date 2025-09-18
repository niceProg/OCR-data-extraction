from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile as StarletteUploadFile
from typing import Dict, Any
import io

from .ocr import run_tesseract_ocr
from .do_client import call_do_gpt4o, extract_assistant_content, DOClientError

app = FastAPI(title="OCR + DigitalOcean GPT-4o Demo")

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "upload"
    content = await file.read()
    try:
        extracted_text, meta = run_tesseract_ocr(content)
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": f"OCR failed: {e}", "filename": filename},
        )

    # Show result and provide option to call GPT-4o
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": filename,
            "extracted_text": extracted_text,
            "meta": meta,
        },
    )


@app.post("/clean", response_class=HTMLResponse)
async def clean_with_model(request: Request, extracted_text: str = Form(...)):
    """
    Receives extracted_text from form, calls DO GPT-4o to clean + structure the text.
    """
    # Build prompt (tweak as needed)
    user_prompt = f"""
You are given OCR-extracted raw text delimited by triple backticks. Clean OCR artifacts (fix hyphenated line breaks,
merge broken words, remove repeated whitespace). Detect the language and produce:
1) A clean_text field (string) - cleaned full text.
2) lines - array of important lines (split logically).
3) summary - 1-2 sentence summary.
4) possible_fields - try to detect structured items like date, invoice_no, total, currency, name if present.

Return ONLY a valid JSON object.

Here is the text:
"""
    system_msg = (
        "You are an assistant that returns a JSON object only — no extra commentary."
    )

    try:
        resp_json = call_do_gpt4o(user_prompt, system=system_msg)
        assistant_text = extract_assistant_content(resp_json)
    except DOClientError as e:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": str(e), "extracted_text": extracted_text},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": f"Model call failed: {e}",
                "extracted_text": extracted_text,
            },
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "extracted_text": extracted_text,
            "model_response": assistant_text,
        },
    )
