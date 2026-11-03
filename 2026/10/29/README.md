# ⚡ FastAPI 비동기/동기 아키텍처 & 고신뢰성 서빙(Serving) 가이드

본 문서는 FastAPI 기반 AI 모델 서빙 환경에서 **이벤트 루프 블로킹을 방지하고, CPU/GPU 및 I/O 특성에 맞게 동기/비동기/스레드풀을 올바르게 아키텍처링**하여 고가용성 무중단 서비스를 구축하기 위한 실무 표준 핸드북입니다.

---

## 📑 목차 (Table of Contents)
1. [비동기(Async) vs 병렬(Parallel)의 본질과 번역의 함정](#1-비동기async-vs-병렬parallel의-본질과-번역의-함정)
2. [하드웨어 코어와 소프트웨어 스레드 (내 CPU의 비밀)](#2-하드웨어-코어와-소프트웨어-스레드-내-cpu의-비밀)
3. [FastAPI 동시 처리 3대 패턴 실전 벤치마크 (11초 vs 5초)](#3-fastapi-동시-처리-3대-패턴-실전-벤치마크-11초-vs-5초)
4. [헬스체크(`/health`)와 쿠버네티스(K8s) 생존 원리](#4-헬스체크health와-쿠버네티스k8s-생존-원리)
5. [실무 판별 공식: `def` vs `async def` 마스터 테이블](#5-실무-판별-공식-def-vs-async-def-마스터-테이블)
6. [글로벌 에러 핸들러 & 프로덕션 로깅 (`print` 퇴출)](#6-글로벌-에러-핸들러--프로덕션-로깅-print-퇴출)
7. [💡 [실무 플러스] 시니어 엔지니어가 반드시 알아야 할 지식 (Bonus Pro-Tips)](#7--실무-플러스-시니어-엔지니어가-반드시-알아야-할-지식-bonus-pro-tips)

---

## 1. 비동기(Async) vs 병렬(Parallel)의 본질과 번역의 함정

교재의 *"비동기 = 동시 실행"*이라는 거친 번역은 **동시성(Concurrency)**과 **병렬성(Parallelism)**을 혼동하게 만듭니다.

```text
[ 1. 동기 (Sync / Blocking) ]   ➔ 요리사 1명이 물 끓을 때까지 멍하니 서서 기다림 (시간 낭비)
[ 2. 비동기 (Async / Non-blocking) ] ➔ 요리사 1명이 물 끓는 동안 다른 손님 양파를 썸 (대기 시간 제거)
[ 3. 병렬 처리 (Parallelism) ]  ➔ 주방에 물리적인 요리사가 실제로 3명 있어서 각자 볶음 (다중 코어)
```

### 🧺 손빨래(CPU) vs 세탁기(I/O)의 법칙
* **CPU/GPU 바운드 연산 (손빨래):**  
  내 손(CPU)이 100% 묶여서 3초 동안 일해야 하므로, **일꾼 1명이 3개 하면 무조건 9초($3 \times 3$) 소요!** 비동기로 돌려도 빨라지지 않음.
* **I/O 바운드 대기 (세탁기):**  
  CPU는 시작 신호(버튼 누르기, 0.001초)만 던지고 빠지며, 실제 일은 바깥(세탁기/외부 네트워크/DB)에서 함. **3대 동시에 버튼 누르고 기다리니 3초 만에 3개 완료!**

---

## 2. 하드웨어 코어와 소프트웨어 스레드 (내 CPU의 비밀)

```text
[ 컴퓨터 본체 ]
   ├── 🧠 CPU 코어 (Core) : 주방에 서 있는 '물리적인 요리사' (하드웨어 쇳덩이)
   ├── 🏠 프로세스 (Process) : 독립된 '식당 매장' (메모리 공간 완전 격리)
   └── 🧵 스레드 (Thread) : 그 매장 안에서 주방을 공유하며 일하는 '직원의 작업 흐름'
```

* **인텔 i5-12400F의 반전:**
  * `12400`의 12는 코어 수가 아니라 **12세대**를 의미.
  * **진짜 물리 코어: 6개** | **논리 스레드(하이퍼스레딩): 12개**
  * 파이썬 `os.cpu_count()`가 12를 뱉은 이유는 하드웨어 쇳덩이가 아니라 **OS 눈에 보이는 '스레드(일꾼) 수'**를 세었기 때문.

---

## 3. FastAPI 동시 처리 3대 패턴 실전 벤치마크 (11초 vs 5초)

3초 걸리는 동기 작업 3개를 동시에 요청했을 때의 실측 결과:

```text
[실험 결과 요약]
- 버전 1 (async def + time.sleep)     : 11.0초 (이벤트 루프 동결 대참사 ❌)
- 버전 2 (일반 def, FastAPI 자동)     :  5.1초 (공용 스레드풀로 병렬 처리 ⭕)
- 버전 3 (async def + run_in_executor) :  5.0초 (전담 스레드풀 외주 위임 ✅ 실무 권장)
```

### ① 버전 1의 비극 (`async def` 안의 동기 코드)
* `async def`는 메인 이벤트 루프 단일 스레드에서 돌아감.
* 그 안에서 `time.sleep`이나 무거운 딥러닝 연산을 돌리면 **메인 이벤트 루프 뇌세포가 굳어버려 1번 손님(11초) ➔ 2번 손님(8초) ➔ 3번 손님(5초)으로 1열 종대 대기열이 누적됨.**

### ② 버전 2의 한계 (일반 `def`)
* 일반 `def`로 쓰면 FastAPI가 알아서 공용 스레드풀로 보내 5.1초로 단축해 줌.
* **한계:** FastAPI 공용 스레드풀은 **기본 정원이 약 40명**임. 손님 40명이 동시에 무거운 PyTorch 추론을 요청하면 **GPU VRAM이 폭발(`CUDA OOM`)하며 서버가 다운될 위험 존재.**

### ③ 버전 3의 우수성 (`run_in_executor` - 개발자의 정밀 통제)
```python
# 1. AI 전담 알바생을 딱 4명으로 통제 (GPU OOM 방어선)
inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")

@app.post("/predict/v3-executor")
async def predict_v3():
    loop = asyncio.get_event_loop()
    # 2. 카운터(지배인)는 비동기로 손님을 받고, 무거운 요리만 전담팀에 외주!
    result = await loop.run_in_executor(inference_executor, heavy_inference)
    return result
```
* **자동 최적화가 아니라 개발자의 수동 설계:**  
  * 입구 카운터(`async def`)는 수천 명의 가벼운 요청을 안 멈추고 받아냄.
  * 무거운 GPU 롤러코스터는 `max_workers=4` 안전 게이트를 통과시켜 **딱 4명씩만 태워 GPU OOM을 원천 방어!**

---

## 4. 헬스체크(`/health`)와 쿠버네티스(K8s) 생존 원리

헬스체크는 AWS 로드밸런서와 쿠버네티스가 **5초마다 서버의 심장박동을 확인하는 생명선**입니다.

```text
[ blocking 버전 (async def + sleep) ]
POST /predict 완료 (5.0초) ➔ 그 뒤에야 GET /health 완료 (4.5초 대기!)
🚨 대참사: 헬스체크가 2초 넘게 묵묵부답이라 쿠버네티스가 "서버 죽었다"며 컨테이너 강제 재부팅(Kill)!

[ threadpool / executor 버전 ]
GET /health 완료 (즉시 통과!) ➔ POST /predict 완료 (뒤에서 5.0초 동안 천천히 끝남)
✅ 생존: 무거운 AI 추론이 돌고 있어도 카운터 지배인이 0.001초 만에 헬스체크에 "정상!" 즉답.
```

---

## 5. 실무 판별 공식: `def` vs `async def` 마스터 테이블

판단의 절대 기준: **"코드에 `await`를 붙여 딴짓하러 갈 수 있는가?"**

| 작업 유형 | 추천 선언 | 선택 이유 |
| :--- | :---: | :--- |
| **로컬 딥러닝 추론 (PyTorch, ONNX)** | **`def`** (또는 `run_in_executor`) | 내 컴퓨터 GPU/CPU를 100% 혹사하므로 스레드풀에 외주 위임 |
| **외부 거대 모델 호출 (OpenAI, Claude API)** | **`async def`** | 인터넷 선 타고 돌아올 때까지 대기하는 I/O이므로 `await` 활용 |
| **MCP (Model Context Protocol)** | **`async def`** | 슬랙, 깃허브, 외부 툴과의 네트워크 통신(I/O)이므로 비동기 필수 |
| **RAG 파이프라인 (Vector DB + LLM)** | **`async def`** | Qdrant 검색(네트워크) + LLM 호출(네트워크) 대기 시간이 대부분 |
| **웹소켓 / 실시간 스트리밍 (SSE)** | **`async def`** | 수만 명과의 연결을 열어두어야 하므로 스레드 낭비 없는 비동기 필수 |
| **대용량 이미지 전처리 (Pillow 4K Resize)** | **`def`** | 830만 개 픽셀을 CPU로 행렬 계산함. `await`가 없으므로 무조건 스레드풀 위임 |
| **전통적인 사내 DB (psycopg2, 기본 SQLAlchemy)** | **`def`** | 라이브러리가 비동기를 지원 안 하므로 `async`에 넣으면 이벤트 루프 멈춤 |

---

## 6. 글로벌 에러 핸들러 & 프로덕션 로깅 (`print` 퇴출)

### ① `print()` vs `logger` 비교

| 비교 항목 | `print()` | `logger.error()` |
| :--- | :--- | :--- |
| **성격** | 개발자 개인의 포스트잇 낙서장 | 사고 났을 때 법원에 제출하는 공문서 (블랙박스) |
| **메타데이터** | 텍스트만 출력 | **날짜, 초 단위 시간, 에러 등급(ERROR), 모듈명 자동 부착** |
| **운영 통제** | 코드 다 뒤져서 일일이 지워야 함 | 설정 한 줄(`level="WARNING"`)로 불필요한 로그 일괄 음소거 |
| **저장/알림** | 터미널 꺼지면 영구 증발 | **로그 파일 영구 보존, 슬랙 알림, Datadog/Sentry 자동 전송** |
| **동시성 안전** | 스레드 여러 개가 치면 글자 깨짐 | **스레드 안전(Thread-safe)하여 줄 서서 깔끔 출력** |

### ② 프로덕션 에러 핸들링 철칙 (보안 분리)
```python
@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    # 1. 서버 내부 블랙박스에는 상세한 족보(traceback)를 전부 기록 (개발자 디버깅용)
    logger.error(f"에러: {exc}\n스택 트레이스:\n{traceback.format_exc()}")
    
    # 2. 바깥 손님(해커)에게는 안전하고 차가운 한 줄만 반환 (사내 코드 유출 방지)
    return JSONResponse(status_code=500, content={"success": False, "error": "서버 내부 오류가 발생했습니다."})
```

---

## 7. 💡 [실무 플러스] 시니어 엔지니어가 반드시 알아야 할 지식 (Bonus Pro-Tips)

### Tip 1. `run_in_executor`의 최신 파이썬 문법: `asyncio.to_thread`
파이썬 3.9부터는 복잡한 `loop = asyncio.get_event_loop()` 없이 **한 줄로 스레드풀에 넘기는 현대적 문법**을 지원합니다:
```python
# Python 3.9+ 최신 스타일 (FastAPI 권장)
import asyncio

result = await asyncio.to_thread(heavy_inference)
```

### Tip 2. 파이썬 GIL의 한계와 진짜 CPU 연산을 위한 `ProcessPoolExecutor`
* `ThreadPoolExecutor`는 `time.sleep`이나 C++로 짜인 PyTorch/ONNX 연산 중에는 파이썬 락(GIL)이 풀려서 병렬로 잘 돕니다.
* 하지만 **순수 파이썬 반복문(수학 계산, 복잡한 문자열 파싱)**은 파이썬 GIL 때문에 스레드를 10개 띄워도 코어 1개에서 돕니다.
* 이때는 스레드가 아니라 프로세스를 통째로 복제하는 **`ProcessPoolExecutor` (멀티 프로세싱)**를 써야 진짜 물리적 CPU 코어를 다 쓰게 됩니다.

### Tip 3. FastAPI 기본 스레드풀 정원(40개) 조정법
일반 `def`를 쓸 때 FastAPI의 기본 스레드풀(AnyIO) 정원은 40개입니다. 서버 사양이 아주 좋아서 알바생 정원을 늘리고 싶다면 앱 시작 시 조절할 수 있습니다:
```python
from anyio.to_thread import current_default_thread_limiter

@app.on_event("startup")
def adjust_thread_limit():
    # 기본 40개 ➔ 100개로 증원!
    current_default_thread_limiter().total_tokens = 100
```

---

### 🏆 오늘 달성한 엔지니어링 마일스톤
* ✅ 비동기(Concurrency)와 병렬(Parallelism)의 개념적 차이 완벽 정립
* ✅ CPU 코어와 스레드, 하이퍼스레딩 동작 원리 이해
* ✅ FastAPI 3대 동시 처리 패턴(V1/V2/V3) 벤치마크 검증
* ✅ K8s 헬스체크 생존을 위한 블로킹 방어 아키텍처 습득
* ✅ 실무 5대 시나리오별 `def` vs `async def` 의사결정 기준 완성
* ✅ 프로덕션 수준의 로깅 및 보안 에러 핸들러 구축 완료!