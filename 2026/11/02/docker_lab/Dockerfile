FROM python:3.11-slim
WORKDIR /app

# [권장 표준] 1. 패키지 목록만 먼저 복사해 무거운 pip install 레이어를 캐싱하고 --no-cache-dir로 용량을 줄입니다
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 자주 바뀌는 애플리케이션 코드는 맨 뒤에 복사합니다
COPY app/ ./app/

ENV MODEL_VERSION="1.0.0"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
