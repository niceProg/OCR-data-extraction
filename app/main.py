from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.do_client import clean_ocr_text

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process", response_class=HTMLResponse)
async def process_text(request: Request, raw_text: str = Form(...)):
    # langsung bersihkan pakai GPT-4o
    cleaned = clean_ocr_text(raw_text)
    return templates.TemplateResponse(
        "result.html", {"request": request, "raw": raw_text, "cleaned": cleaned}
    )
