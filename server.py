"""WebSocket API сервер для Multilingual STT"""

import json
import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from src.config import SUPPORTED_LANGUAGES, SAMPLE_RATE
from src.asr.model import ASRModel
from src.translation.translator import Translator
from src.abbreviations.handler import AbbreviationHandler
from src.postprocessing.punctuation import PunctuationRestorer
from src.postprocessing.spelling import SpellingCorrector

app = FastAPI(title="Multilingual STT API")

# Глобальні моделі (завантажуються один раз)
models = {}


def get_models():
  """Lazy-завантаження моделей"""
  if not models:
      print("Loading models...")
      models["asr"] = ASRModel(lang="ukr", device="cpu")
      models["translator"] = Translator(device="cpu")
      models["abbreviations"] = AbbreviationHandler()
      models["spelling"] = SpellingCorrector()
      models["punctuation"] = PunctuationRestorer()
      print("All models loaded!")
  return models


@app.get("/")
async def root():
  """Проста тестова сторінка"""
  return HTMLResponse("""
  <html><head><title>STT Server</title></head>
  <body>
      <h1>Multilingual STT Server</h1>
      <p>WebSocket endpoint: ws://localhost:8000/ws</p>
      <p>Languages: """ + ", ".join(
          f"{k} ({v['name']})" for k, v in SUPPORTED_LANGUAGES.items()
      ) + """</p>
  </body></html>
  """)


@app.get("/health")
async def health():
  return {"status": "ok", "languages": list(SUPPORTED_LANGUAGES.keys())}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  await websocket.accept()
  m = get_models()

  # Конфігурація сесії
  source_lang = "uk"
  target_lang = "uk"

  try:
      while True:
          data = await websocket.receive_text()
          msg = json.loads(data)

          if msg["type"] == "config":
              source_lang = msg.get("source_lang", "uk")
              target_lang = msg.get("target_lang", "uk")
              lang_config = SUPPORTED_LANGUAGES[source_lang]
              m["asr"].set_language(lang_config["mms_code"])
              await websocket.send_json({
                  "type": "config_ok",
                  "source_lang": source_lang,
                  "target_lang": target_lang,
              })

          elif msg["type"] == "audio":
              # Аудіо приходить як список float
              audio = np.array(msg["audio"], dtype=np.float32)

              # Розпізнавання
              raw_text = m["asr"].transcribe(audio)

              if not raw_text.strip():
                  await websocket.send_json({
                      "type": "partial",
                      "text": "",
                      "source_lang": source_lang,
                  })
                  continue

              is_final = msg.get("is_final", False)

              if is_final:
                  # Post-processing
                  text = m["abbreviations"].process(raw_text)
                  text = m["spelling"].correct(text)
                  text = m["punctuation"].restore(text)

                  # Переклад
                  translated = m["translator"].translate(
                      text, source_lang, target_lang
                  )

                  await websocket.send_json({
                      "type": "final",
                      "text": translated,
                      "original": raw_text,
                      "source_lang": source_lang,
                      "target_lang": target_lang,
                  })
              else:
                  await websocket.send_json({
                      "type": "partial",
                      "text": raw_text,
                      "source_lang": source_lang,
                  })

  except WebSocketDisconnect:
      print("Client disconnected")


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)