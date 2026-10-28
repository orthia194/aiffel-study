"""
자동 문서를 닫은 서버 (운영 서버의 모양)
"""
from fastapi import FastAPI

app = FastAPI(
    title="My First ML API",
    docs_url=None,       # /docs 를 만들지 않습니다
    redoc_url=None,      # /redoc 을 만들지 않습니다
    openapi_url=None,    # /openapi.json 을 만들지 않습니다
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
