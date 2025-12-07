from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
from tokenization_small100 import SMALL100Tokenizer
from onnx_greedy import greedy_generate_onnx
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class TranslationRequest(BaseModel):
    text: str
    flag: str | None = None

# Загружаем модель
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
tokenizer.tgt_lang = "ru"

encoder_session = ort.InferenceSession("encoder_model_int8.onnx")
decoder_session = ort.InferenceSession("decoder_model_int8.onnx")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(req: TranslationRequest):

    text = req.text
    flag = req.flag

    if not text:
        return JSONResponse({"error": "no text provided"}, status_code=400)

    # Устанавливаем язык
    if flag:
        tokenizer.tgt_lang = flag

    # Генерация
    output_ids = greedy_generate_onnx(
        encoder_session,
        decoder_session,
        tokenizer,
        text,
        max_new_tokens=50
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"translation": output_text}   # <--- ВАЖНО!


