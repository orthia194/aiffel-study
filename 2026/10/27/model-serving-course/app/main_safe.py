"""
422와 500 응답에 request ID를 붙인 서버 (운영 서버의 모양)
"""
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Safe Error API")

LOG_PATH = "server_errors.log"


def write_log(line: str):
    """서버 쪽 기록. 로그를 남기는 정식 방법(logging 모듈)은 비동기 처리 노트에서 배웁니다."""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


# ===== 422 : 검증 실패 응답을 바꿔 끼웁니다 =====
@app.exception_handler(RequestValidationError)
def on_validation_error(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    # 로그에는 위치(loc)와 종류(type)만 남깁니다. 클라이언트가 보낸 원문(input)은 남기지 않습니다
    brief = [(e.get("loc"), e.get("type")) for e in exc.errors()]
    write_log(f"422 {request_id} {request.url.path} {brief}")
    return JSONResponse(
        status_code=422,
        content={"detail": {"error": "요청 형식이 올바르지 않습니다.", "request_id": request_id}},
    )


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


def run_model(text: str) -> dict:
    """모델 자리. 공백만 있는 글은 토큰이 0개라 추론 중 예외가 납니다."""
    tokens = text.split()
    if not tokens:
        raise RuntimeError("models/sentiment_v1.pth 추론 실패: 토큰이 0개입니다")
    return {"label": "긍정", "confidence": 0.92}


# ===== 500 : 예외 메시지 대신 request ID를 보냅니다 =====
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        return run_model(request.text)
    except Exception as e:
        request_id = uuid.uuid4().hex[:12]
        write_log(f"500 {request_id} /predict {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "추론 중 에러가 났습니다.", "request_id": request_id},
        )
