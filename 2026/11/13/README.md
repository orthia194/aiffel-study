# 🚀 LLM 서빙 엔진 성능 벤치마크: vLLM vs TensorRT-LLM

동일한 조건과 공정한 측정 기준 하에서 **vLLM**과 **TensorRT-LLM**의 추론 성능(지연 시간·처리량·SLO 충족 여부)을 비교하고, 실무 환경에 맞는 서빙 엔진 및 GPU 선택 기준을 정리한 실습 기록입니다.

---

## 📌 TL;DR (핵심 요약)

1. **공정 비교가 먼저다**: 질문 세트(`seed` 고정·중복 제거), 예열(`warmup`), `ignore_eos`, 동시성 4배 이상 요청 수, KV 캐시 할당량(±20% 이내)을 맞추지 않으면 벤치마크 결과는 쉽게 왜곡됩니다.
2. **엔진 대결 (`NVIDIA L4 / fp16` 기준)**:
   - **저부하(동시 1~8)**: vLLM과 TensorRT-LLM 간 체감 성능 차이가 거의 없습니다.
   - **고부하(동시 32)**: **TensorRT-LLM**이 vLLM 대비 총 처리량(`output tok/s`)에서 **약 14% 우위**(`~1,880` vs `~1,650 tok/s`), `TTFT p99`(`~890` vs `~925 ms`) 및 `TPOT p99`(`~21` vs `~25 ms`)에서도 소폭 앞섭니다.
3. **하드웨어 체급 차이 (`L4` vs `A40`)**:
   - 같은 `vLLM(fp16)`이라도 메모리 대역폭이 2.3배 넓은 `A40 (696 GB/s)`이 `L4 (300 GB/s)` 대비 동시 32 기준 처리량 약 1.6배(`2,665 tok/s`), `TPOT p99`는 절반 이하(`~11 ms`)로 압도적인 차이를 보입니다.
4. **결론**: 특별한 제약이 없다면 **vLLM이 기본값**이며, **① NVIDIA GPU(CC 8.0 이상)이고 ② 모델이 장기간 고정되며 ③ 동시 부하가 높아 GPU 효율을 끝까지 끌어올려야 할 때** 실측 검증 후 TensorRT-LLM 전환을 고려합니다.

---

## 🛠️ 실험 환경 및 벤치마크 설계

### 1. 테스트 환경
| 항목 | 내 실측 환경 | 비교 데이터 (강사 CSV) |
| :--- | :--- | :--- |
| **GPU** | NVIDIA A40 (Ampere, CC `8.6`, 48GB, `696 GB/s`) | NVIDIA L4 (Ada, CC `8.9`, 24GB, `300 GB/s`) |
| **모델** | `Qwen/Qwen2.5-1.5B-Instruct` (`fp16`) | `Qwen/Qwen2.5-1.5B-Instruct` (`fp16`) |
| **최대 길이** | `4096` tokens | `4096` tokens |
| **SLO 목표** | `TTFT p99 ≤ 2000 ms`, `TPOT p99 ≤ 100 ms` | `TTFT p99 ≤ 2000 ms`, `TPOT p99 ≤ 100 ms` |

### 2. 공정성 확보 장치 (Fairness Validation)
엔진별 전용 벤치마크 도구를 따로 쓰지 않고, 동일한 OpenAI 호환 클라이언트(`bench_openai.py`)와 사전 감사 함수(`fairness_report`)를 구현해 비교 오류를 차단했습니다.

* **질문 생성 (`make_prompts.py`)**:
  * `seed=59` 고정으로 600개 질문 생성 (`sha256: 8df0816a0a74f397`)
  * 짧은 문의(70%, 중앙값 63자) + 사내 규정 RAG형 긴 질문(30%, 최대 2,174자) 혼합으로 **Prefill과 Decode가 섞이는 현실 트래픽** 재현
  * 매 질문마다 고유 문의 번호(`[문의 0000]`)와 랜덤 조항 조합을 넣어 **Prefix Cache 적중에 의한 왜곡 방지** (중복 0개)
* **측정 클라이언트 (`bench_openai.py`)**:
  * `stream_options: {"include_usage": True}`로 서버 공식 `completion_tokens`만 신뢰 (SSE 청크 수로 토큰 수를 추정하지 않음)
  * `ignore_eos=True`, `temperature=0.0`, `max_tokens=128` 고정으로 생성 길이 변동 차단
  * 매 동시성 단계마다 `warmup`(4건) 선행 후 `max(16, 4 * concurrency)` 요청 전송
  * `httpx.Limits` 커넥션 풀 상한을 `concurrency + 4` 이상으로 열어 클라이언트 병목 방지
* **자동 감사 (`fairness_report`)**:
  * `MUST_MATCH` 12개 조건 일치 여부 확인 (`model`, `model_revision`, `precision`, `gpu`, `prompt_sha`, `prompt_offset`, `output_tokens`, `ignore_eos`, `gen_params`, `warmup`, `num_requests`, `tool`)
  * 실측 여부(`source == "실측"`), 예열 여부, 요청 수(`>= 4 * concurrency`), 질문 반복(`repeats == 0`), 에러(`errors == 0`), **KV 캐시 토큰 크기 차이 20% 이내** 검증

---

## 📊 실험 결과 및 해석

### 1. NVIDIA A40 / fp16 — vLLM 실측 결과
> 설정: `--dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-chunked-prefill`  
> 로그 확인: 어텐션 백엔드 `FLASH_ATTN` / KV 캐시 `1,267,792 tokens` / 서버 준비 `156.6 s`

| 동시 요청 수 (`concurrency`) | 요청 수 | `TTFT p50` | `TTFT p99` (SLO: 2000ms) | `output tok/s` | 실패 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 16 | 43.5 ms | 77.0 ms | 138.6 tok/s | 0 |
| **2** | 16 | 71.3 ms | 120.4 ms | 265.4 tok/s | 0 |
| **4** | 16 | 120.0 ms | 164.2 ms | 497.7 tok/s | 0 |
| **8** | 32 | 136.5 ms | 220.1 ms | 944.9 tok/s | 0 |
| **16** | 64 | 231.9 ms | 308.0 ms | 1,699.5 tok/s | 0 |
| **32** | 128 | 420.5 ms | 655.0 ms | **2,665.0 tok/s** | 0 |

* **선형 스케일링**: 동시성이 1 → 32로 증가할 때 `TPOT p99`는 `~7ms → ~11ms`로 거의 일정하게 유지되면서, 총 처리량은 `138.6 → 2,665.0 tok/s`(**약 19.2배**)로 증가했습니다. (Continuous Batching 효과)
* **SLO 여유**: 동시 32에서도 `TTFT p99`는 목표(`2000ms`)의 약 1/3, `TPOT p99`는 목표(`100ms`)의 약 1/9 수준으로 넉넉히 통과했습니다.

### 2. NVIDIA L4 / fp16 — vLLM vs TensorRT-LLM 비교
* **동시 1 ~ 8**: 두 엔진의 `TTFT p99`, `TPOT p99`, `output tok/s`가 거의 겹칩니다.
* **동시 32 (고부하)**:
  * `output tok/s`: TensorRT-LLM(`~1,880`) vs vLLM(`~1,650`) → **TensorRT-LLM 약 14% 우위**
  * `TPOT p99`: TensorRT-LLM(`~21 ms`) vs vLLM(`~25 ms`) → **TensorRT-LLM 더 안정적**
  * `TTFT p99`: TensorRT-LLM(`~890 ms`) vs vLLM(`~925 ms`) → **TensorRT-LLM 소폭 우위**

---

## 📏 LLM 서빙에서 가장 중요한 핵심 지표 (Metrics)

실제 LLM 서비스를 운영하거나 벤치마크할 때 반드시 확인해야 하는 지표들입니다.

| 지표 | 의미 및 계산 방식 | 왜 중요한가? |
| :--- | :--- | :--- |
| **TTFT**<br>*(Time To First Token)* | 요청 전송 ~ 첫 출력 토큰 수신 시간<br>(`Queue 대기 시간 + Prefill 계산 시간`) | 사용자가 엔터를 누른 뒤 **"반응이 온다"**고 느끼는 체감 반응성 지표 (낮을수록 좋음) |
| **TPOT**<br>*(Time Per Output Token)* | 첫 토큰 이후 출력 토큰 1개당 평균 생성 간격<br>`(t_last - t_first) / (out_tokens - 1)` | 글자가 타닥타닥 찍히는 **체감 출력 속도** 지표. 역수를 취하면 인당 `tok/s` (낮을수록 좋음) |
| **ITL**<br>*(Inter-Token Latency)* | 연속된 개별 토큰 사이의 **순간 간격** | `TPOT`이 요청 평균이라 스파이크를 가리는 반면, `ITL`은 중간에 Prefill이 끼어들어 **잠깐 멈칫하는 현상**까지 포착함 |
| **E2E Latency**<br>*(End-to-End)* | 요청 전송 ~ 마지막 토큰 수신 완료 시간<br>`TTFT + TPOT × (N - 1)` | 스트리밍이 아닌 **JSON 파싱, 에이전트 툴 호출, 배치 파이프라인** 등 전체 응답이 다 와야 다음 단계로 넘어가는 서비스의 핵심 지표 |
| **Throughput**<br>*(처리량: `tok/s`, `req/s`)* | 서버 전체가 초당 생성하는 출력 토큰(`output tok/s`) 및 완료 요청 수(`RPS`) | GPU 1장당 생산성과 **서빙 비용(토큰당 원가)**을 결정 (높을수록 좋음) |
| **Goodput**<br>*(SLO 충족 처리량)* | **목표 SLO(`TTFT`, `TPOT`)를 어기지 않으면서** 처리할 수 있는 최대 동시 요청 수 / `tok/s` | 단순 최대 처리량보다 실무에서 더 중요! SLO를 초과하는 구간은 실제 서비스에서 쓸 수 없으므로 **오토스케일링 기준선**으로 사용 |
| **Tail Latency**<br>*(`p95`, `p99`)* | 전체 요청 중 가장 느린 상위 5%, 상위 1%의 지연 시간 | 평균(`mean`)이나 중앙값(`p50`)이 좋아도 `p99`가 튀면 일부 사용자 경험이 무너지므로 SLO는 대부분 **`p99`** 기준으로 설정 |
| **KV Cache Usage & Preemption** | VRAM 내 KV 캐시 점유율(%) 및 캐시 부족으로 진행 중 요청이 밀려난(선점/재계산) 횟수 | 동시성이 폭주할 때 `TTFT` 급등과 지연 스파이크의 선행 경고 지표 |

---

## 💡 추가로 알아두면 좋은 서빙 아키텍처 지식

### 1. 왜 GPU마다 `TTFT`와 `TPOT`의 병목이 다를까?
* **Prefill 단계 (→ `TTFT` 결정)**: 입력 프롬프트 토큰 전체를 한 번에 행렬 곱셈하므로 **연산량 병목(Compute-bound)**입니다. GPU의 **TFLOPS**가 중요합니다.
* **Decode 단계 (→ `TPOT` 결정)**: 글자 1개를 뽑을 때마다 수 GB의 모델 가중치를 VRAM에서 읽어와야 하므로 **메모리 대역폭 병목(Memory-Bandwidth-bound)**입니다. GPU의 **대역폭(GB/s)**이 중요합니다.
* 이 물리적 차이 때문에 메모리 대역폭이 2.3배 넓은 A40(`696 GB/s`)이 L4(`300 GB/s`)보다 `TPOT`에서 2배 넘게 빨랐으며, 최근 대규모 서빙에서는 Prefill 서버와 Decode 서버를 아예 분리하는 **Disaggregated Serving** 구조도 활용됩니다.

### 2. FP8 양자화와 GPU 아키텍처 조건
* 이번 실습의 `트랙 1 (FP8)`이 A40에서 건너뛰어진 이유: 하드웨어 **FP8 Tensor Core**는 **Ada Lovelace(CC 8.9, 예: L4/RTX 4090)** 및 **Hopper(CC 9.0, 예: H100)** 이상부터 지원됩니다.
* **Ampere(CC 8.6, A40)**에서는 FP16/BF16이나 INT8/INT4(AWQ, GPTQ) 양자화를 사용해야 합니다.

### 3. 대표 추론 엔진 3종 포지셔닝 (`vLLM` vs `SGLang` vs `TensorRT-LLM`)
| 엔진 | 핵심 특징 및 강점 | 추천 상황 |
| :--- | :--- | :--- |
| **vLLM** | 가장 넓은 모델/하드웨어 호환, 빠른 신모델 지원(Day-0), 쉬운 배포와 운영 | **실무 기본값 (Default)** — 모델이 자주 바뀌거나 빠르게 안정적 서빙을 띄울 때 |
| **SGLang** | `RadixAttention` 기반 토큰 단위 트리형 Prefix 캐시 재사용, 정형 출력(JSON) 강점 | 긴 시스템 프롬프트·대화 맥락이 계속 겹치는 **멀티턴 스토리 챗, RAG, 에이전트** |
| **TensorRT-LLM** | NVIDIA GPU 전용 극한 커널/런타임 최적화 (v1.2+부터는 정적 `.engine` 빌드 없이 PyTorch 백엔드 + YAML 설정으로 `trtllm-serve` 지원) | **CC 8.0+ NVIDIA GPU** 환경에서 **고정 모델**을 고부하·대규모로 돌려 GPU 비용을 10~20%라도 더 아껴야 할 때 |

### 4. 실무에서 엔진 교체 제안 시 체크리스트
> ⚠️ *"TensorRT-LLM은 커널 융합, CUDA 그래프, 인플라이트 배칭을 지원하므로 vLLM보다 빠를 것이다"* 같은 주장은 근거가 불충분합니다. **Continuous(In-flight) Batching, CUDA Graphs, Kernel Fusion(FlashAttention 등)은 현재 vLLM도 모두 기본 지원**하기 때문입니다.

엔진 교체를 검토할 때는 기능표가 아니라 아래 4가지를 확인합니다:
1. **하드웨어**: GPU가 NVIDIA **Ampere(CC 8.0) 이상**인가?
2. **운영 주기**: 모델 교체가 잦지 않고 장기간 고정 운영하는가?
3. **실익(ROI / Goodput)**: 현재 vLLM으로 SLO를 못 맞추거나, GPU 클러스터 규모가 커서 10~20% 처리량 향상이 실질적 비용 절감으로 이어지는가?
4. **가성비($/1M tokens)**: 절대 속도가 빠른 상위 GPU(예: A40)와 전성비 GPU(예: L4) 중 **시간당 비용 대비 토큰 생산량**이 유리한 쪽은 어디인가?

---

## ▶️ 재현 방법 (Quick Start)

```bash
# 1. 공정 벤치마크용 프롬프트 생성 (600개, 중복 0)
python make_prompts.py

# 2. vLLM 서버 실행 예시 (FP16)
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --generation-config vllm \
  --port 8000

# 3. 동일 조건 OpenAI API 벤치마크 실행
python bench_openai.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --engine vllm \
  --precision fp16 \
  --prompts d59_prompts.jsonl \
  --concurrency 1,2,4,8,16,32 \
  --out d59_bench_mine.csv
```