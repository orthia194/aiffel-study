# 🚀 FastAPI 기반 고성능 머신러닝 모델 서빙(Serving) 아키텍처 가이드

본 문서는 PyTorch/ONNX 기반 딥러닝 모델을 **FastAPI, Uvicorn, Pydantic**을 활용해 엔터프라이즈급 실시간 RESTful API 서버로 서빙·검증·운영하기 위한 실무 표준 핸드북입니다.

---

## 📑 목차 (Table of Contents)
1. [서빙 생태계의 핵심 삼총사 (Uvicorn, Pydantic, FastAPI)](#1-서빙-생태계의-핵심-삼총사-uvicorn-pydantic-fastapi)
2. [파이썬 패키지 구조와 `__init__.py`의 비밀](#2-파이썬-패키지-구조와-__init__py의-비밀)
3. [RESTful API 3대 파라미터 완전 정복 (Path, Query, Body)](#3-restful-api-3대-파라미터-완전-정복-path-query-body)
4. [자동 문서화 (Swagger UI vs ReDoc) & 운영 보안 강화](#4-자동-문서화-swagger-ui-vs-redoc--운영-보안-강화)
5. [HTTP 상태 코드와 실무 장애 대응 (422 vs 500/502/504)](#5-http-상태-코드와-실무-장애-대응-422-vs-500502504)
6. [실전 딥러닝(CNN) 이미지 서빙 엔드투엔드 파이프라인](#6-실전-딥러닝cnn-이미지-서빙-엔드투엔드-파이프라인)
7. [💡 [실무 플러스] 시니어 엔지니어의 핵심 팁 (Bonus Pro-Tips)](#7--실무-플러스-시니어-엔지니어의-핵심-팁-bonus-pro-tips)

---

## 1. 서빙 생태계의 핵심 삼총사 (Uvicorn, Pydantic, FastAPI)

현대 파이썬 AI 백엔드는 **식당의 역할 분담**처럼 3개의 기술이 유기적으로 맞물려 돌아갑니다.

```text
[ 👤 클라이언트 (손님) ]
       │ (HTTP 주문서 요청)
       ▼
[ 🚪 1. Uvicorn (문지기 & 배달원) ] ➔ 포트(8000)를 열고 네트워크 패킷(HTTP 바이트) 수신
       │
       ▼
[ 🕵️‍♂️ 2. Pydantic (입구 세관원) ]   ➔ 데이터 타입과 길이를 0.001초 만에 검수 (불량은 422 컷!)
       │ (검수 통과 데이터)
       ▼
[ 👨‍💼 3. FastAPI (총지배인) ]       ➔ 주방의 AI 모델(PyTorch/ONNX)을 호출하고 결과를 JSON 서빙
```

* **Uvicorn (ASGI 웹 서버):** 파이썬 코드 바깥에서 실제 8000번 포트를 열고 클라이언트와 통신하는 초고속 비동기 엔진.
* **Pydantic (데이터 검증):** 모델이 이상한 데이터를 먹고 배탈(GPU OOM, 차원 에러)나지 않도록 문앞에서 타입과 유효성을 철벽 검증.
* **FastAPI (웹 프레임워크):** URL 라우팅, AI 추론 로직 오케스트레이션, 대화형 문서 자동 생성을 총괄하는 지휘자.

---

## 2. 파이썬 패키지 구조와 `__init__.py`의 비밀

빈 파일로 존재하는 `app/__init__.py`는 단순한 빈 껍데기가 아닌 중요한 역할을 수행합니다.

1. **패키지 신분증:**  
   파이썬에게 해당 디렉토리가 단순 잡동사니 폴더가 아니라, 코드를 점(`.`) 찍어 가져올 수 있는 **정식 파이썬 패키지**임을 선언.
2. **도구 호환성 (pytest, Docker):**  
   이 파일이 없으면 테스트 도구(`pytest`)나 배포 툴이 모듈 경로를 못 찾아 `ModuleNotFoundError`를 유발하는 것을 방지.
3. **재수출(Re-export / Facade 패턴):**  
   내부의 깊은 경로를 1층 로비(안내 데스크)에서 한 번에 노출시켜 외부 호출 코드를 단순화:
   ```python
   # app/__init__.py 에 미리 작성:
   from .model_utils import load_model, predict
   
   # 외부에서는 파일명 몰라도 초간단 임포트:
   from app import load_model, predict
   ```

---

## 3. RESTful API 3대 파라미터 완전 정복 (Path, Query, Body)

FastAPI는 함수 인자의 선언 방식에 따라 파라미터의 위치를 자동으로 판별합니다.

| 구분 | 문법 형태 | 실제 주소 예시 | 주 사용 목적 |
| :--- | :--- | :--- | :--- |
| **Path 파라미터** | `@app.get("/models/{model_name}")` | `/models/sentiment-v1` | **단 하나의 고유한 대상(ID, 이름)**을 콕 집어 조회할 때 |
| **Query 파라미터** | `def list_models(status: str = None, limit: int = 10)` | `/models?status=running&limit=10` | **목록 검색, 필터링, 정렬, 페이징** (주소 공유 & 뒤로가기 가능) |
| **Request Body** | `def predict(req: PredictRequest)` (`BaseModel` 상속) | (주소창에 안 보임, HTTP 본문 전송) | **이미지 픽셀, 긴 문장, 복잡한 딕셔너리 등 대용량 데이터 전송** |

### ① Query 파라미터의 실무 규칙: `limit: int = 10`
* `limit`: 변수명 (가져올 데이터 상한선)
* `: int`: Pydantic 타입 힌트 (문자열 입력 시 422 에러 차단)
* `= 10`: 손님이 값을 안 적으면 자동으로 10개만 슬라이싱(`models[:limit]`)하여 **데이터 폭주로부터 서버 메모리 보호.**

### ② `response_model`의 출력 필터링 (보안 장치)
```python
@app.post("/predict", response_model=PredictResponse)
```
* 내부 딕셔너리에 실수로 사내 기밀 변수나 불필요한 키가 들어있더라도, **`PredictResponse`에 정의된 필드만 쏙 뽑아서 내보내므로 데이터 유출 원천 방지.**
* 선택 필드(`probabilities: Optional[dict] = None`)가 생략되면 알아서 `null`로 채워 출력 규격 일관성 보장.

---

## 4. 자동 문서화 (Swagger UI vs ReDoc) & 운영 보안 강화

FastAPI는 작성된 코드와 Pydantic 모델을 파싱하여 **`/openapi.json` 단일 명세서**를 생성하고, 이를 2가지 뷰어로 제공합니다.

| 비교 항목 | **Swagger UI (`/docs`)** | **ReDoc (`/redoc`)** |
| :--- | :--- | :--- |
| **성격** | 대화형 인터랙티브 놀이터 (체험형 키오스크) | 우아한 3단 분할 공식 책자 (열람용 메뉴판) |
| **테스트** | **[Try it out] 버튼으로 브라우저 즉석 테스트 가능 (Postman 대체)** | 테스트 불가 (순수 읽기 전용) |
| **주 사용처** | 내부 개발자, 프론트엔드 동료와의 실시간 협업 | 외부 고객사 납품, 공식 개발자 센터 공개용 |

### 🔒 운영(Production) 서버 보안 하드닝 (Security Hardening)
운영 서버에 `/docs`를 열어두면 해커에게 친절한 공격 설계도(Blueprint)를 쥐여주는 꼴이 됩니다.  
따라서 **상용 배포 시에는 3대 경로를 전부 `None`으로 봉쇄**해야 합니다:
```python
import os
from fastapi import FastAPI

IS_PROD = os.getenv("ENVIRONMENT") == "production"

app = FastAPI(
    docs_url=None if IS_PROD else "/docs",        # Swagger UI 차단
    redoc_url=None if IS_PROD else "/redoc",      # ReDoc 차단
    openapi_url=None if IS_PROD else "/openapi.json" # 원시 JSON 뒷문까지 차단!
)
```

---

## 5. HTTP 상태 코드와 실무 장애 대응 (422 vs 500/502/504)

### ① 왜 422(Unprocessable Entity)는 "천사 에러"인가?
* **100% 클라이언트 잘못!** Pydantic이 서버 내부로 데이터가 들어가기 전에 문앞에서 컷.
* 족집게 과외 수준으로 원인을 명시(`"loc": ["body", "text"], "msg": "Field required"`).
* 백엔드 서버는 안전하며, 프론트엔드에게 요청 수정만 요구하면 되므로 개발자의 야근을 유발하지 않음.

### ② 개발자의 피를 말리는 "악마의 3대 에러" (5xx)
1. **`500 Internal Server Error` (서버 내부 폭발):**  
   * 코드 미처리 예외, GPU CUDA OOM(메모리 부족), 텐서 차원 불일치로 서버가 다운. 긴급 로그 분석 필요.
2. **`502 Bad Gateway` (주방장 사망):**  
   * Nginx가 Uvicorn에 접근했으나 프로세스가 흔적도 없이 꺼져있음. 리눅스 커널의 OOM Killer가 프로세스를 강제 종료한 경우가 많아 파이썬 에러 로그조차 안 남음.
3. **`504 Gateway Timeout` (AI 서빙의 고질병):**  
   * LLM이 60초 넘게 답변을 생성하느라 Nginx의 타임아웃 제한을 초과. (스트리밍 도입 필수).

### ③ 상태 코드 커스텀 규칙 (505 / 606의 위험성)
* 상태 코드는 전 세계 표준(RFC 100~599)이므로, `606` 같은 비표준 번호를 임의로 쓰면 Nginx/AWS 게이트웨이가 비정상 응답으로 판단해 연결을 끊어버림.
* **실무 정석:** 겉봉투(상태 코드)는 `503` 표준을 지키고, 회사만의 고유 에러는 **본문 JSON(`detail`)**에 기록:
  ```python
  raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY_606", "msg": "..."})
  ```

---

## 6. 실전 딥러닝(CNN) 이미지 서빙 엔드투엔드 파이프라인

$1 \times 28 \times 28$ MNIST 숫자 분류 모델 서빙 구현 핵심.

### ① 데이터 계약 및 검증 (`schemas.py`)
```python
class PredictRequest(BaseModel):
    pixel_values: list[float] = Field(..., min_length=784, max_length=784)
    return_probabilities: bool = False
```
* 783개나 785개가 들어오면 파이토치가 `reshape` 하다가 500 에러를 뿜기 전에, **Pydantic이 문앞에서 `422 에러`로 즉시 입구 컷.**

### ② 안전한 싱글톤 모델 로딩 (`main.py`)
* 요청(Request)마다 모델을 로드하면 매번 수 초 지연 및 메모리 폭발 발생.
* **서버 기동 시 모듈 레벨에서 딱 1번만 메모리에 로드**하여 모든 요청이 공유:
  ```python
  try:
      model = load_model("models/mnist_state_dict.pth")
      model_loaded = True
  except Exception:
      model_loaded = False
  ```

### ③ 텐서 차원 변환 및 추론
* JSON 리스트(784개) ➔ CNN 4차원 텐서 변환:
  ```python
  input_tensor = torch.tensor(request.pixel_values, dtype=torch.float32).reshape(1, 1, 28, 28)
  ```
* 전처리 정규화(`Normalize((0.1307,), (0.3081,))`)가 학습 환경과 정확히 일치하여 **연속 10개 실전 테스트에서 10/10 (정확도 100%, 확신도 99%+) 달성!**

---

## 7. 💡 [실무 플러스] 시니어 엔지니어의 핵심 팁 (Bonus Pro-Tips)

### Tip 1. `.pt` vs `.pth`의 오해 종결
* `.pt`와 `.pth`는 기능적 차이가 없는 **개발자들 사이의 암묵적 관례(Convention)**일 뿐.
* 둘 다 가중치만 담을 수도 있고, 모델 전체를 담을 수도 있음.
* **실무 표준:** 보안 위험(Pickle 코드 실행)과 파일 이동 깨짐을 방지하기 위해 확장자 상관없이 **무조건 `state_dict`(가중치 딕셔너리)만 저장하고 `weights_only=True`로 로드하는 것이 절대 원칙.**
* *(참고: 클래스 코드 없이 단독 실행 가능한 파일은 ONNX 또는 TorchScript 컴파일 파일)*

### Tip 2. 27x27 이미지의 운명 (Pydantic vs 전처리기)
* **픽셀 리스트 직접 전송 시:** $27 \times 27 = 729$개이므로 Pydantic이 사진인 줄 모르고 개수 부족으로 **422 에러 발생.**
* **강제로 크게(Upscale) 하려면?**  
  서버가 파일(`UploadFile`)을 받도록 설계하고, 내부에서 `transforms.Resize((28, 28))`를 적용하면 컴퓨터 비전의 **보간법(Interpolation)**이 없는 픽셀을 주변 색상으로 채워 28로 뻥튀기해 줌!

### Tip 3. 대용량 이미지 처리 기법 (Crop vs Resize)
* 거대한 영수증/문서 사진에서 숫자를 인식할 때:
  1. **관심 영역 자르기 (Crop / ROI):** YOLO/OpenCV로 숫자가 있는 사각형(Bounding Box) 위치만 쏙 오려내어 모델에 전달.
  2. **리사이징 (Resize):** 전체 사진을 가로세로 28픽셀 크기의 썸네일로 압축 축소.

---

### 🏆 오늘 달성한 엔지니어링 마일스톤
* ✅ Uvicorn + Pydantic + FastAPI 아키텍처 원리 완벽 이해
* ✅ Path, Query, Body 3대 파라미터 핸들링 및 유효성 검증
* ✅ Swagger UI / ReDoc 활용 및 운영 보안(하드닝) 적용
* ✅ HTTP 상태 코드(422, 500, 502, 504) 기반 에러 제어
* ✅ PyTorch CNN 모델과 연동한 실시간 추론 API 구축 및 100% 적중률 검증 완료!