from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
import numpy as np
from tokenization_small100 import SMALL100Tokenizer
from onnx_greedy import greedy_generate_onnx

app = FastAPI()

# Шаблоны (templates/index.html)
templates = Jinja2Templates(directory="templates")


# Загружаем ONNX модель (файл должен лежать рядом: app/model.onnx)
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
tokenizer.tgt_lang = "ru"
#################################
encoder_session = ort.InferenceSession("encoder_model_int8.onnx")
decoder_session = ort.InferenceSession("decoder_model_int8.onnx")





@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    text = payload.get("text", "")
    if not text:
        return JSONResponse({"error": "no text provided"}, status_code=400)

    # Подготовка входов через токенизатор (возвращаем numpy массивы)
    output_ids = greedy_generate_onnx(
        encoder_session,
        decoder_session,
        tokenizer,
        text,
        max_new_tokens=50
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Преобразуем выход (обычно logits/ids) в удобный формат.
    # Здесь просто вернём raw-первый выход как список.
    return {"result": output_text}
