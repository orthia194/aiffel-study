# 🚀 High-Throughput LLM Serving & Production MLOps Pipeline

> **vLLM과 FastAPI를 활용한 고성능 LLM 추론 가속, 양자화(AWQ), 용량 산정 및 2-Tier 서빙 아키텍처 구축 실습**

본 프로젝트는 고성능 LLM 추론 최적화 기법(**PagedAttention, GQA, Continuous Batching, Chunked Prefill, FlashAttention**)의 동작 원리를 이해하고, 직접 4비트 AWQ 양자화 모델을 빌드하여 단일 GPU 기준 동시 수용량을 극대화한 실무 서빙 파이프라인 구축 과정을 담고 있습니다.

---

## 📌 목차
1. [핵심 이론 및 최적화 아키텍처](#1-핵심-이론-및-최적화-아키텍처)
2. [하드웨어 환경 및 벤치마크 실험 분석](#2-하드웨어-환경-및-벤치마크-실험-분석)
3. [수학적 모델링: KV 캐시 용량 산정 (Capacity Planning)](#3-수학적-모델링-kv-캐시-용량-산정-capacity-planning)
4. [2-Tier 프로덕션 서빙 아키텍처 (FastAPI + vLLM)](#4-2-tier-프로덕션-서빙-아키텍처-fastapi--vllm)
5. [💡 MLOps 엔지니어가 반드시 알아야 할 추가 지식](#5--mlops-엔지니어가-반드시-알아야-할-추가-지식)

---

## 1. 핵심 이론 및 최적화 아키텍처

### ① PagedAttention (vLLM)
* **문제점:** 요청마다 시퀀스 길이가 달라 전통적인 방식에서는 최대 길이에 맞춰 연속된 물리 메모리를 과다 할당(Pre-allocation)하여 60~80%의 심각한 메모리 단편화 발생.
* **해결 원리:** OS의 가상 메모리 페이징 기법을 도입. KV 캐시를 **고정 블록(16토큰 단위)**으로 잘게 쪼개어 물리 메모리 곳곳에 불연속적으로 동적 할당하고, Page Table로 논리 매핑.
* **결과:** 메모리 단편화 낭비를 **4% 미만**으로 절감하여 동시 처리량 대폭 향상.

### ② GQA (Grouped-Query Attention)
* **문제점:** MHA(Multi-Head Attention)는 $N$개의 Query마다 $N$개의 Key/Value를 가져와야 해 디코딩 시 메모리 대역폭 병목(Memory-bound) 발생.
* **해결 원리:** MQA(1개 KV 공유)의 품질 저하와 MHA의 메모리 낭비를 절충. 여러 Query 헤드를 그룹으로 묶어 그룹당 1쌍의 Key/Value 헤드를 공유 ($64Q : 8K : 8V$).
* **결과:** MHA 대비 KV 캐시 크기를 **1/8로 축소**하여 대역폭 병목을 해소하고 최신 LLM(Llama 3, Qwen 2.5)의 표준으로 자리 잡음.

### ③ 연속 배칭 (Continuous Batching / Iteration-level Batching)
* **문제점:** 정적 배칭(Static Batching)은 배치 내 가장 긴 요청이 끝날 때까지 짧은 요청이 무의미한 패딩(`<pad>`)을 채우며 GPU가 유휴(Idle) 상태가 됨.
* **해결 원리:** 요청 단위가 아닌 **토큰 생성 스텝(Iteration) 단위**로 스케줄링. 토큰 생성이 끝난 요청은 즉시 퇴출(Evict)하고 대기열의 새 요청을 즉시 투입(Insert).
* **결과:** GPU 연산 유휴 시간을 제거하여 Throughput(처리량)을 극대화.

### ④ 청크 프리필 (Chunked Prefill)
* **문제점:** 긴 프롬프트 입력(Prefill, Compute-bound)이 들어오면 GPU를 장시간 독점하여, 기존 사용자들의 토큰 생성(Decode, Memory-bound) 흐름이 순간적으로 뚝 끊기는 렉(Jitter) 발생.
* **해결 원리:** 긴 질문을 일정 크기(예: 512토큰 단위 조각)로 쪼개어, 각 스텝마다 **[프리필 청크 + 기존 요청의 디코드 작업]**을 섞어서(Piggybacking) 동시 처리.
* **결과:** 응답 지연 시간(P99 Latency) 안정화 및 GPU 연산기(Tensor Core)와 메모리 대역폭을 100% 동시 가동.

### ⑤ FlashAttention & AWQ (Activation-aware Weight Quantization)
* **FlashAttention:** 느린 VRAM(HBM)과 빠른 온칩 SRAM 사이의 I/O 병목을 해결하기 위해 타일링(Tiling)과 온라인 소프트맥스를 적용, SRAM 안에서 어텐션 계산을 완결.
* **AWQ:** 활성화값(Activation)의 크기를 관찰하여 가장 중요한 1%의 가중치를 보호하며 4비트로 양자화(W4A16). 메모리 읽기량을 획기적으로 줄여 가속 달성.

---

## 2. 하드웨어 환경 및 벤치마크 실험 분석

### 환경 스펙 비교
* **Colab Environment:** NVIDIA A100 SXM4 (80GB VRAM, HBM2e ~2,000 GB/s 대역폭)
* **RunPod Environment:** NVIDIA A40 (48GB VRAM, GDDR6 ~696 GB/s 대역폭)

### 📊 6대 서빙 실험군 비교 (RunPod A40 기준)

| 실험군 | 모델 | 설정값 | 모델 가중치 | KV 캐시 토큰 | 동시 수용량 | 단독 속도 | 최대 지연 | 정답 수 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. A 기준** | Qwen-2.5-32B-AWQ | util 0.92 / len 4096 | 18.14 GiB | 82,704 | 654명 | 15.4 t/s | 1.31s | 26/30 |
| **2. A 비율 낮춤** | Qwen-2.5-32B-AWQ | **util 0.65** | 18.14 GiB | 38,144 | 301명 | 15.4 t/s | 1.31s | 26/30 |
| **3. B max_len 확장**| Qwen-2.5-32B-AWQ | **len 32768** | 18.14 GiB | 82,704 | 2.52x | 15.6 t/s | 1.27s | 26/30 |
| **4. C 배칭 제한** | Qwen-2.5-32B-AWQ | **seqs 4** | 18.14 GiB | 86,160 | 681명 | 14.2 t/s | **2.48s** | 26/30 |
| **5. D 7B 원본** | Qwen-2.5-7B-Instruct | FP16 원본 | 14.29 GiB | 447,168 | 3,537명 | 22.9 t/s | 0.86s | 23/30 |
| **6. D 7B AWQ (🏆)** | **7b-awq-ko (자체 빌드)**| **4-bit AWQ** | **5.29 GiB** | **633,248** | **5,009명** | **47.4 t/s**| **0.39s** | **26/30** |

#### 💡 핵심 인사이트
1. **자체 구축 7B AWQ 모델의 압도적 성과:**
   * 모델 용량을 14.29GB ➡️ **5.29GB(63% 감량)**로 줄여 남은 VRAM을 KV 캐시로 환원, **동시 수용 인원 5,009명(전체 1위)** 달성.
   * 메모리 대역폭 병목을 제거하여 단독 속도 **47.4 tok/s(32B 대비 3배 이상)** 기록.
   * 32B 지식 증류/한국어 캘리브레이션 효과로 16비트 원본(23개)보다 높은 **26개 정답(32B급 지능)** 획득.
2. **배칭 제한(`seqs=4`)의 위험성:**
   * 동시 처리 슬롯을 4개로 제한하자 대기 큐(`Waiting`)가 11개까지 치솟으며 **최대 지연 시간이 2.48초로 2배 폭증**.
3. **컨텍스트 길이(`len=32768`)의 함정:**
   * 무리하게 컨텍스트를 늘리면 vLLM이 보수적으로 메모리를 예약하여 동시 처리 배율이 **20.19x에서 2.52x로 폭락**.

---

## 3. 수학적 모델링: KV 캐시 용량 산정 (Capacity Planning)

### ① 토큰당 KV 캐시 메모리 공식
$$\text{Memory per Token} = 2 (\text{Key, Value}) \times \text{Layers} \times \text{KV Heads} \times \text{Head Dim} \times \text{Precision (Bytes)}$$

* **Qwen-2.5-32B (GQA 적용: 64 Layers, 8 KV Heads, Dim 128, FP16):**
  $$2 \times 64 \times 8 \times 128 \times 2\text{ bytes} = 262,144\text{ bytes} = \mathbf{256\text{ KiB / token}}$$
* **블록 단위 크기 (Block size = 16):**
  $$256\text{ KiB} \times 16 = \mathbf{4\text{ MiB / block}}$$
* **전체 가용 KV 풀:**
  $$13,333\text{ blocks} \times 4\text{ MiB} = \mathbf{52.08\text{ GiB}}$$

### ② 동시 수용 인원 및 SLO 분석 (`TARGET_SECONDS = 6.0s`)
* **평균 요청 길이:** 질문 + 답변 합계 평균 약 126토큰 (요청당 평균 7.9개 블록 소모)
* **VRAM 기준 수용 한계 ($N$):** $82,704 \div (7.9 \times 16) \approx \mathbf{654\text{명}}$
* **응답 시간(SLO) 기준 한계 ($M$):** 부하 테스트(동시 30명) 결과 최대 지연 **1.12초**로 6초 기준을 여유롭게 통과 $\rightarrow \mathbf{M \ge 30\text{명}}$
* **Hugging Face 순차 처리 비교:** 단일 0.26초 소요 $\rightarrow 6.0초 \div 0.26초 \approx \mathbf{23\text{명}}$ (24번째 요청부터 SLO 위반)

---

## 4. 2-Tier 프로덕션 서빙 아키텍처 (FastAPI + vLLM)

GPU 자원을 독점하는 추론 엔진과 클라이언트 인터페이스를 담당하는 API 게이트웨이를 완전히 분리하여 **확장성, 안정성, 보안성**을 확보했습니다.

```text
[클라이언트 앱 (Web / Mobile)]
           │
           ▼  (X-API-Key 헤더 인증, HTTP POST)
┌────────────────────────────────────────────────────────┐
│  Tier 1: FastAPI Gateway Microservice (app/chatbot_api)│
│  - 인증 관리: auth.py (401 Unauthorized 방어)          │
│  - 스키마 검증: chatbot_schemas.py (Pydantic, 422 방어) │
│  - 보안 래핑: errors.py (내부 Traceback 은닉, 500 방어)  │
│  - 과부하 제어(Backpressure): ThreadPoolExecutor       │
│    (MAX_WORKERS = VLLM_LIMIT = 30 으로 수치 일치)      │
└────────────────────────────────────────────────────────┘
           │
           ▼  (사내 내부 네트워크 /v1/chat/completions)
┌────────────────────────────────────────────────────────┐
│  Tier 2: vLLM Inference Engine (vllm serve)            │
│  - PagedAttention + FlashAttention 백엔드 가동          │
│  - AWQ Marlin 고속 커널 연산                           │
│  - 프로메테우스 메트릭 수집 (/metrics)                   │
└────────────────────────────────────────────────────────┘
```

### 🧪 프로덕션 규격 8개 검증 항목 올패스 (8 / 8 Pass)
* [x] 1. `/health` 헬스체크 정상 작동 (200 OK)
* [x] 2. 단일 턴 질의 규격 준수 (200 OK, Response Key 포맷 검증)
* [x] 3. 멀티 턴 맥락 기억 답변 성공 (200 OK)
* [x] 4. API Key 누락 요청 차단 (401 Unauthorized)
* [x] 5. 비인가 API Key 위변조 차단 (401 Unauthorized)
* [x] 6. 빈 메시지 페이로드 거부 (422 Unprocessable Entity)
* [x] 7. Temperature 초과값 거부 (422 Unprocessable Entity)
* [x] 8. 고의 GPU 에러 유발 시 코드 경로 및 스택 트레이스 은닉 (500 Error Sanitization)

---

## 5. 💡 MLOps 엔지니어가 반드시 알아야 할 추가 지식

### ① 지연 시간 지표: TTFT vs TPOT
실무에서 LLM 서빙 SLA를 정의할 때 '평균 지연 시간'만 보면 안 됩니다.
* **TTFT (Time to First Token):** 사용자가 질문을 던지고 **첫 글자가 화면에 찍힐 때까지의 시간** (프리필 속도, Chunked Prefill 여부에 직결).
* **TPOT (Time Per Output Token):** 첫 글자가 나온 후 **글자가 한 자 한 자 찍히는 스트리밍 속도** (디코딩 속도, 메모리 대역폭에 직결).
* *운영 팁:* 사용자는 TTFT가 1초 이내이고 TPOT이 초당 20토큰 이상이면 렉이 없다고 체감합니다.

### ② 프롬프트 캐싱 (Prefix Caching)
시스템 프롬프트가 길거나(예: 3,000자 규정집), 동일한 Few-shot 예시를 반복 전달하는 서비스라면 vLLM의 `--enable-prefix-caching`을 활성화해야 합니다. 공통 접두사의 KV 캐시 블록을 재사용하여 프리필 연산량을 **최대 90%까지 스킵**할 수 있습니다.

### ③ 투기적 디코딩 (Speculative Decoding)
디코딩 속도를 극적으로 올리는 기법입니다. 작은 초고속 모델(Draft Model, 예: 0.5B)이 다음 토큰 5개를 빠르게 추측해서 생성하고, 메인 모델(Target Model, 예: 32B)이 한 번의 검증(Forward Pass)으로 이를 채택/기각합니다. 품질 저하 없이 디코딩 속도를 **1.5배~2.5배 가속**합니다.

### ④ 프로메테우스 모니터링 경보(Alert) 임계치
실무 쿠버네티스(K8s) 환경에서 HPA(자동 스케일아웃)를 걸거나 슬랙 알람을 울려야 하는 기준:
1. `vllm:num_requests_waiting > 0`: GPU 용량이 꽉 차서 손님이 줄을 서기 시작함 ➡️ 즉시 GPU 파드 증설 필요.
2. `vllm:kv_cache_usage_perc > 0.85`: KV 메모리 85% 돌파 ➡️ OOM 직전이므로 새 요청 차단(Rate Limiting) 또는 스케일아웃.

---

## 🛠️ 기술 스택 (Tech Stack)
* **Model:** Qwen/Qwen2.5-32B-Instruct-AWQ, Qwen/Qwen2.5-7B-Instruct (Custom AWQ 4-bit)
* **Serving Engine:** vLLM (`vllm serve`, PagedAttention, Continuous Batching)
* **Backend Framework:** FastAPI, Uvicorn, Pydantic v2
* **Acceleration Kernels:** FlashAttention-2, Marlin Linear Kernel
* **Infrastructure:** PyTorch, RunPod (NVIDIA A40 48GB), Docker, HTTPX
