import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Depends, HTTPException, status
from app.auth import verify_api_key
from app.schemas import PredictRequest, PredictResponse
from app.model_service import load_model, predict

app = FastAPI(
    title="한국어 감정 분석 API 서비스",
    description="KoELECTRA 모델 기반 감정 분석 서빙 API",
    version="1.0.0"
)

model = None
executor = ThreadPoolExecutor(max_workers=2)

@app.on_event("startup")
async def startup_event():
    global model
    print("🚀 [FastAPI] 감정 분석 모델을 로드 중입니다...")
    try:
        model = load_model()
        print("✅ [FastAPI] 모델 로드 완료!")
    except Exception as e:
        print(f"❌ [FastAPI] 모델 로드 실패: {e}")
        model = None

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def run_predict(
    request: PredictRequest,
    user: str = Depends(verify_api_key)  # 인증 실패 시 auth.py에서 401 반환
):
    # 1. 모델이 준비되지 않은 경우 (503 Service Unavailable)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 아직 준비되지 않았거나 로드에 실패했습니다."
        )

    # 2. 비동기 스레드 풀에서 추론 수행 및 모델 예외 처리 (500 Internal Server Error)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, predict, model, request.text)
        result["user"] = user  # 인증된 사용자 정보 포함
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추론 처리 중 모델 오류가 발생했습니다: {str(e)}"
        )
