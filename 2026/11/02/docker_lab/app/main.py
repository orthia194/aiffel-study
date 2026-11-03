import os, json, time, asyncio, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
API_KEY = os.environ.get("SERVICE_API_KEY", "dev-secret-key")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/sentiment_weights.json"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "serving.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger("private_llm_api")

# GPU/CPU 추론 동시 실행 슬롯을 통제하는 전용 스레드풀 (D48 핵심 패턴)
inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infer")

def load_weights():
    if MODEL_PATH.exists():
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    # 단일 컨테이너 기초 실습(볼륨 미마운트 시)을 위한 기본 가중치 폴백
    return {"model_name": "builtin-fallback", "pos_weights": {"좋다": 1.0, "안정": 1.0}, "neg_weights": {"오류": -1.0, "에러": -1.0}, "bias": 0.0}

MODEL_ARTIFACT = load_weights()

app = FastAPI(title="Private Model Serving API", version=MODEL_VERSION)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="분석할 한국어 문장")

def run_sync_inference(text: str) -> dict:
    score = MODEL_ARTIFACT.get("bias", 0.0)
    for w, weight in MODEL_ARTIFACT["pos_weights"].items():
        if w in text:
            score += weight
    for w, weight in MODEL_ARTIFACT["neg_weights"].items():
        if w in text:
            score += weight
    label = "positive" if score >= 0 else "negative"
    return {"label": label, "score": round(score, 3), "model_source": MODEL_ARTIFACT["model_name"]}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_source": MODEL_ARTIFACT["model_name"],
        "weights_mounted": MODEL_PATH.exists()
    }

@app.post("/predict")
async def predict(req: PredictRequest, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="유효하지 않은 API Key입니다.")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(inference_executor, run_sync_inference, req.text)
    logger.info(f"predict_ok label={result['label']} score={result['score']} source={result['model_source']}")
    return {**result, "model_version": MODEL_VERSION}
# code modified
