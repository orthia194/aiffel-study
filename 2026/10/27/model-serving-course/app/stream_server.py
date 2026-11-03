"""LLM API 서버 연습 : 생성 직렬화, SSE 스트리밍, 취소, 예산, 준비 상태

모델은 가짜입니다. 토큰 하나를 만드는 데 걸리는 시간만 흉내 냅니다.
진짜 LLM으로 바꾸는 자리는 generate_tokens() 하나입니다.
"""
import asyncio
import json
import os
import time
from contextlib import aclosing, asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.error_handlers import register_error_handlers
from app.logger_config import setup_logger

logger = setup_logger("ml_api")

# 상수를 코드에 직접 적지 않고 환경 변수에서 읽습니다
TOKENS_PER_SECOND = float(os.environ.get("D48_TPS", "20"))      # 생성 속도
ANSWER_TOKENS = int(os.environ.get("D48_OUT_TOKENS", "60"))   # 답 하나의 토큰 수 (약 3초)
LOAD_SECONDS = float(os.environ.get("D48_LOAD_SECONDS", "3"))    # 모델을 올리는 시간
GPU_SLOTS = int(os.environ.get("D48_GPU_SLOTS", "1"))            # 동시에 생성하는 요청 수
MAX_TOKENS = int(os.environ.get("D48_MAX_TOKENS", "500"))        # 토큰 예산 기본값
WALL_CLOCK = float(os.environ.get("D48_WALL_CLOCK", "30"))       # 시간 예산 기본값 (초)

STATE = {"준비됨": False}
STATS = {"만든 토큰": 0}
gpu_limiter = anyio.CapacityLimiter(GPU_SLOTS)


def generate_tokens(question):
    """가짜 LLM. 토큰을 하나씩 내놓는 동기 제너레이터입니다"""
    for i in range(ANSWER_TOKENS):
        time.sleep(1 / TOKENS_PER_SECOND)        # 토큰 하나를 계산하는 시간
        STATS["만든 토큰"] += 1
        yield f"토큰{i} "


async def load_model():
    await asyncio.sleep(LOAD_SECONDS)             # 진짜 서버라면 여기서 모델을 GPU에 올립니다
    STATE["준비됨"] = True
    logger.info("모델 준비 완료")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(load_model())      # 뜰 때 : 모델 올리기를 뒤에서 시작합니다
    yield                                          # 이 사이에 요청을 받습니다
    task.cancel()                                  # 내릴 때 : 정리합니다
    logger.info("서버를 내립니다")


app = FastAPI(title="LLM Streaming API", lifespan=lifespan)
register_error_handlers(app)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    max_tokens: int | None = Field(default=None, ge=1, le=2000)          # 이 요청의 토큰 예산
    deadline_seconds: float | None = Field(default=None, gt=0, le=300)   # 이 요청의 시간 예산


def new_budget(req):
    """요청이 예산을 주지 않으면 서버 기본값을 씁니다"""
    return {"max_tokens": req.max_tokens or MAX_TOKENS,
            "deadline": req.deadline_seconds or WALL_CLOCK}


async def guarded_tokens(question, request, budget, report):
    """한 번에 GPU_SLOTS 건만 통과시키고, 예산과 클라이언트 이탈을 보며 토큰을 내놓습니다"""
    it = generate_tokens(question)
    done = object()
    count = 0
    queued_at = started = time.time()
    report["정지 사유"] = "밖에서 끊김"
    try:
        async with gpu_limiter:                    # 슬롯이 빌 때까지 여기서 기다립니다
            started = time.time()
            report["대기 초"] = round(started - queued_at, 2)
            while True:
                # next() 한 번(토큰 하나)만 스레드에서 돌립니다
                token = await anyio.to_thread.run_sync(next, it, done)
                if token is done:
                    report["정지 사유"] = "완료"
                    break
                count += 1
                yield token
                if count >= budget["max_tokens"]:
                    report["정지 사유"] = "토큰 예산 소진"
                    break
                if time.time() - started >= budget["deadline"]:
                    report["정지 사유"] = "시간 예산 소진"
                    break
                if await request.is_disconnected():
                    report["정지 사유"] = "클라이언트 이탈"
                    break
    finally:
        it.close()                                 # 생성을 실제로 멈춥니다
        report["토큰"] = count
        report["생성 초"] = round(time.time() - started, 2)
        logger.info(f"생성 종료 : {report}")


def sse(event, data):
    """SSE 한 청크. 빈 줄 하나로 끝납니다"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def not_ready():
    return JSONResponse(status_code=503, content={"error": "모델을 올리는 중입니다"})


@app.get("/health")
async def health():
    return {"status": "healthy" if STATE["준비됨"] else "loading"}


@app.get("/stats")
async def stats():
    return dict(STATS)


@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    """답을 다 만든 뒤 한 번에 돌려줍니다"""
    if not STATE["준비됨"]:
        return not_ready()
    report, parts = {}, []
    async with aclosing(guarded_tokens(req.question, request, new_budget(req), report)) as stream:
        async for token in stream:
            parts.append(token)
    return {"answer": "".join(parts), **report}


@app.post("/ask/sse")
async def ask_sse(req: AskRequest, request: Request):
    """만들어지는 대로 SSE로 스트리밍합니다"""
    if not STATE["준비됨"]:
        return not_ready()
    report = {}

    async def events():
        yield sse("start", {"question": req.question})
        try:
            async with aclosing(guarded_tokens(req.question, request, new_budget(req), report)) as stream:
                async for token in stream:
                    yield sse("token", {"text": token})
        except Exception as exc:
            # 200이 이미 나갔으므로 오류도 본문 안의 청크로 알립니다
            logger.error(f"생성 중 오류 : {exc}")
            yield sse("error", {"error": "생성 중 오류가 발생했습니다"})
            return
        yield sse("done", report)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})   # 앞단 프록시가 모아 두지 않게
