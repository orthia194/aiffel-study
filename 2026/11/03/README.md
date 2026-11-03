# 🚀 Production-Ready ML/LLM Serving Pipeline & Gateway Architecture

본 프로젝트는 모델 학습부터 인프라 배포까지, 실제 엔터프라이즈 환경에서 무장애(Zero-Downtime)와 안정적인 서빙을 달성하기 위한 **엔드투엔드(End-to-End) 서빙 파이프라인** 및 **게이트웨이 아키텍처**를 다룹니다.

---

## 📌 목차
1. [시스템 아키텍처 및 파이프라인 개요](#1-시스템-아키텍처-및-파이프라인-개요)
2. [핵심 질문: 모델 가중치 외에 무엇을 함께 동반 저장해야 하는가?](#2-핵심-질문-모델-가중치-외에-무엇을-함께-동반-저장해야-하는가)
3. [파이프라인 구현 상세](#3-파이프라인-구현-상세)
4. [프로덕션 서빙 & 게이트웨이 핵심 원칙](#4-프로덕션-서빙--게이트웨이-핵심-원칙)
5. [테스트 및 검증 결과](#5-테스트-및-검증-결과)

---

## 1. 시스템 아키텍처 및 파이프라인 개요

```text
[ Client (Streamlit / Web) ]
             │ (HTTP POST /predict)
             ▼
[ API Gateway (Reverse Proxy) ]
   - 인증 검문 (401 Unauthorized)
   - 트래픽 속도 제어: 토큰 버킷 알고리즘 (429 Too Many Requests)
   - 부하 분산: 최소 연결(Least Connections) 기반 라우팅
   - 분산 추적: Trace ID (X-Request-ID) 발급
             │
             ▼
[ FastAPI Serving Engine ]
   - 입구 컷 유효성 검사: Pydantic Schema (422)
   - 이벤트 루프 블로킹 방지: run_in_executor (Worker Thread)
   - 전처리 및 추론: HousingPredictor (피처 순서 보장, Z-Score 정규화)
             │
             ▼
[ PyTorch Regression Model ]
   - 3-Layer MLP (Linear → ReLU → Dropout(0.2) → Linear(32, 1))
   - model.eval() & torch.no_grad() 기반 경량 추론
```

---

## 2. 핵심 질문: 모델 가중치 외에 무엇을 함께 동반 저장해야 하는가?

> **중심 질문:** 학습한 모델을 API 서버와 화면으로 연결할 때, `.pth` 가중치 파일만 있으면 서빙이 가능할까?  
> **결론:** **절대 불가능하다.** 가중치 단독 배포 시 **Train-Serve Skew(학습-서빙 불일치)**로 인해 모델이 엉뚱한 값을 출력한다.

### 동반 저장 및 맞춰야 할 3대 요소
1. **학습 데이터 전처리 통계값 ($\mu, \sigma$):**
   * 사용자는 화면에서 정규화되지 않은 날 것(Raw)의 데이터(예: 소득 $3.5, 연식 25년)를 입력합니다.
   * 모델은 평균 0, 분산 1로 스케일링된 텐서를 기대하므로, 학습 시 사용한 평균(Mean)과 표준편차(Std)를 `housing_preprocessing.json`으로 동반 저장하여 서빙 시 동일하게 Z-Score 변환을 적용해야 합니다.
2. **피처 순서 보장 (Feature Ordering):**
   * API 클라이언트가 전달하는 JSON 딕셔너리는 Key-Value 순서가 보장되지 않습니다.
   * 저장된 `feature_names` 배열을 기준으로 리스트 컴프리헨션을 수행하여 텐서의 열 순서가 섞이지 않도록 강제해야 합니다.
3. **도메인 가드레일 (Domain Guardrail):**
   * 회귀 모델이 극단적인 이상치에 대해 음수 집값을 뱉지 못하도록 `price = max(price, 0.0)`와 같은 비즈니스 규칙을 서빙 모듈에 내장해야 합니다.

---

## 3. 파이프라인 구현 상세

### ① PyTorch 회귀 모델 (`HousingModel`)
* **구조:** `Linear(8, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)`
* **특징:** 회귀 문제이므로 최종 출력층에 Sigmoid/Softmax를 적용하지 않고 실수(Scalar) 그대로 출력.

### ② 추론 코어 캡슐화 모듈 (`HousingPredictor`)
```python
class HousingPredictor:
    def __init__(self, model_path: str, preprocessing_path: str):
        # 1. 전처리 통계 로드
        with open(preprocessing_path, "r") as f:
            params = json.load(f)
        self.mean = np.array(params["mean"])
        self.std = np.array(params["std"])
        self.feature_names = params["feature_names"]

        # 2. 모델 로드 및 추론 모드 전환 (보안/안정성 표준 준수)
        self.model = HousingModel(input_dim=len(self.feature_names))
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()  # Dropout 비활성화 필수

    def predict(self, features: dict) -> dict:
        # 피처 순서 일치 보장
        values = [features[name] for name in self.feature_names]
        
        # 학습 통계 기반 정규화 (Train-Serve Skew 방지)
        normalized = (np.array(values, dtype=np.float32) - self.mean) / self.std

        # 단일 배치 차원 확장 후 추론
        input_tensor = torch.FloatTensor(normalized).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)

        price = max(output.item(), 0.0)  # 음수 방지 가드레일
        return {
            "predicted_price": round(price, 4),
            "predicted_price_usd": int(price * 100000),
        }
```

### ③ FastAPI 비동기 서빙 엔드포인트
* **이벤트 루프 격리:** 무거운 PyTorch 연산이 FastAPI의 메인 이벤트 루프를 블로킹하지 않도록 `run_in_executor`를 사용하여 스레드 풀에 위임.
* **입력 유효성 검사:** Pydantic 스키마의 `Field(ge=..., le=...)`를 통해 비정상적인 범위(음수 방 개수 등)를 모델 인입 전 `422`로 사전 차단.

---

## 4. 프로덕션 서빙 & 게이트웨이 핵심 원칙

### A. 게이트웨이 트래픽 제어 & 상태 코드
| HTTP 코드 | 명칭 | 주 원인 | 서버 상태 | 대처 방식 |
| :---: | :---: | :--- | :---: | :--- |
| **`401`** | Unauthorized | API Key / Token 부재 또는 위조 | 정상 | 게이트웨이 문앞에서 즉시 차단 (Fail-Fast) |
| **`429`** | Too Many Requests | 토큰 버킷 쿼터 초과 (Rate Limit) | **정상** | 클라이언트에게 `Retry-After` 헤더 전달 후 대기 유도 |
| **`503`** | Service Unavailable | 서버 큐 포화, CPU/메모리 과부하, 부팅 중 | **장애/비정상** | 서킷 브레이커 동작, 서버 증설(Scale-out) |
| **`502`** | Bad Gateway | 백엔드 프로세스 Crash (OOM 등) | 장애 | 컨테이너 자동 재시작 및 로그 조사 |

* **토큰 버킷(Token Bucket) 파라미터:**
  * `capacity`: 일시적 트래픽 급증을 허용하는 **순간 최대 버스트 허용량 (Burst Size)**.
  * `refill_per_sec`: 시스템이 장기적으로 감당 가능한 **지속 평균 처리 속도 (Sustained Rate)**.
* **부하 분산(Load Balancing):** 추론 시간 편차(1s~10s)가 큰 AI 서빙 환경에서는 라운드로빈 대신 **최소 연결(Least Connections)** 방식을 적용하여 병목을 방지.

### B. 헬스체크(`/health`) 설계의 2대 금기사항
1. **절대 실제 모델 추론(토큰 생성)을 넣지 말 것:**
   * 피크 타임에 헬스체크가 실제 유저 요청과 GPU 슬롯을 두고 경쟁하다 타임아웃(2초 초과)을 유발함.
   * 게이트웨이가 멀쩡한 서버를 '사망'으로 판단하여 라우팅 풀에서 퇴출시키는 **도미노 전멸 장애** 발생.
   * *해결책:* 서버 시작 시(`lifespan`) 웜업 1회만 수행하고, `/health`는 메모리 플래그(`is_ready = True`)만 즉시 반환.
2. **슬롯 포화(Busy) 상태를 Unhealthy로 보고하지 말 것:**
   * 서버가 바쁜 것(Busy)과 죽은 것(Dead)은 다름. 바쁘다고 Unhealthy를 반환하면 트래픽이 남은 서버로 연쇄 폭탄처럼 이동하여 클러스터 전체가 다운됨.
   * *해결책:* 모델 로딩 중에는 `HTTP 200` + `body: {"status": "loading"}` 형태로 **Readiness**만 분리 표기.

### C. 도커 인프라 & 배포 안정성
* **빌드 레이어 캐시 최적화:** 변경이 드문 `COPY requirements.txt` 및 `RUN pip install`을 앞단에 배치하고, 자주 변경되는 `COPY app/`을 뒤에 두어 빌드 속도 극대화.
* **보안 시크릿 분리:** `RUN rm -f .env`는 하위 레이어에 데이터가 영구 잔류하므로 금지. `.dockerignore` 및 Secret Mount 사용.
* **블루-그린 배포 롤백:** 배포 후 구버전(Blue) 인프라를 즉시 삭제하지 않고 웜 상태로 대기시켜야 라우팅 스위칭만으로 1초 만에 롤백 가능.
* **카나리 배포 (사용자 ID 해시):** 멀티턴 대화형 챗봇에서 턴마다 버전이 바뀌는 현상을 방지하기 위해, 사용자 ID 해시를 기준으로 트래픽을 분할하여 **세션 일관성(Sticky Session)** 유지.
* **사내 폐쇄망(에어갭) 대응:** 인터넷 다운로드를 시도하는 `fetch_*` 데이터셋 및 오픈소스 가중치는 외부망에서 사전 다운로드 후 사내 망연계 반입 절차(`~/scikit_learn_data/`)를 거쳐 배포.

---

## 5. 테스트 및 검증 결과

### 추론 API 테스트 (`POST /predict`)
* **요청 페이로드:**
```json
{
  "MedInc": 3.5,
  "HouseAge": 25.0,
  "AveRooms": 5.0,
  "AveBedrms": 1.0,
  "Population": 1500.0,
  "AveOccup": 3.0,
  "Latitude": 37.5,
  "Longitude": -122.0
}
```

* **서버 응답 (`200 OK`):**
```json
{
  "success": true,
  "predicted_price": 1.8355,
  "predicted_price_usd": 183548,
  "input_features": {
    "MedInc": 3.5,
    "HouseAge": 25.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1500.0,
    "AveOccup": 3.0,
    "Latitude": 37.5,
    "Longitude": -122.0
  }
}
```

* **모델 평가 지표:**
  * Test MSE: `0.3219`
  * Test MAE: `0.3918` (실제 금액 환산: **약 $39,181**)
  * 단일 샘플 테스트 오차 약 $19,060 수준으로 평균 오차 범위 내에서 안정적인 예측 성능 달성.

---

이 문서는 실무 ML/LLM 서빙 환경에서 발생하는 인프라 병목과 장애를 사전에 방지하도록 설계된 표준 구현 가이드라인입니다.