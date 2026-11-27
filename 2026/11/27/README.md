# 📘 LLM Observability & RAG Evaluation Lab (OpenTelemetry + Arize Phoenix)

로컬 Docker(CPU) 환경에서 3단 마이크로서비스 기반의 사내 규정 RAG 시스템을 직접 띄우고, **OpenTelemetry 분산 추적**과 **Arize Phoenix 기반 정량·정성 평가(Eval)** 및 **오류·회귀 분석**을 수행한 실습 정리입니다.

---

## 1. 시스템 아키텍처 & 포트 구성

```text
[Client / Phoenix Experiment Runner]
          │  (HTTP POST /v1/ask, optional traceparent)
          ▼
   ┌─────────────┐    traceparent     ┌──────────────┐    traceparent     ┌─────────────────┐
   │   gateway   │ ─────────────────▶ │  rag-server  │ ─────────────────▶ │   gen_server    │
   │   (:8693)   │                    │   (:8692)    │                    │    (:8690)      │
   └─────────────┘                    └──────────────┘                    └─────────────────┘
          │                                  │                                     │
          │  signals.jsonl                   │  rag.jsonl (질문/답변 원문 분리)        │
          ▼                                  ▼                                     ▼
   [Local JSONL (spans/*.jsonl)] ◀──── d69_otel.py (Dual Export) ────▶ [Arize Phoenix (:8694)]
                                                                       OTLP/HTTP (/v1/traces)
```

| 서비스 / 컴포넌트 | 포트 | 역할 | 사용 모델 / 비고 |
| :--- | :--- | :--- | :--- |
| **gen-a (`gen_server.py`)** | `8690` | OpenAI 호환 추론 서버 | `Qwen/Qwen2.5-0.5B-Instruct` (CPU, `do_sample=False`) |
| **rag-server (`rag_server.py`)** | `8692` | 임베딩·권한 필터링 검색·프롬프트 조립 | `intfloat/multilingual-e5-small` (Top-K=3) |
| **gateway (`gateway.py`)** | `8693` | 요청 진입점, `X-Request-ID` 발급, 피드백 수집 | `/v1/ask`, `/v1/feedback` (`signals.jsonl`) |
| **Arize Phoenix** | `8694` / `8695` | Trace 수집(OTLP HTTP/gRPC), Dataset/Experiment UI | 웹 UI: `http://localhost:8694` |

---

## 2. 핵심 개념 정리

### ① 분산 추적 (Distributed Tracing & OpenTelemetry)
* **W3C `traceparent` 전파 (`TraceContextTextMapPropagator`):** `gateway → rag-server → gen_server`로 이어지는 HTTP 호출마다 `inject` / `extract`로 컨텍스트를 넘겨, 서로 다른 프로세스에서 발생한 작업을 하나의 `trace_id` 트리 아래로 묶습니다.
* **Span Tree 조립 원리:** 각 스팬은 `trace_id`, `span_id`, `parent_id`를 가지며, 루트 스팬은 `parent_id not in {trace 내 span_id 집합}` 조건으로 찾습니다(외부에서 `traceparent`를 이어받은 경우에도 깨지지 않게 처리).
* **이중 출력 (`SimpleSpanProcessor` vs `BatchSpanProcessor`):**
  * 로컬 JSONL 파일(`spans/*.jsonl`): 실험 직후 즉시 읽기 위해 동기식(`SimpleSpanProcessor`)으로 기록.
  * Phoenix OTLP 전송: 서비스 지연을 줄이기 위해 비동기 배치(`BatchSpanProcessor`)로 전송.
* **원문(PII)과 트레이스 분리:** 질문·답변 원문은 트레이스 속성에 직접 싣지 않고 `content/rag.jsonl`에 `trace_id`를 키로 분리 저장해 보안을 유지합니다.

### ② RAG 평가의 3축 분리 (Fault Isolation)
단순 정답률 하나만 보면 *"검색이 틀린 건지, 모델이 지시를 안 따른 건지, 추론을 못 한 건지"* 알 수 없으므로 평가기를 3개로 분리합니다.

| 평가기 | 검사 대상 | 핵심 설계 포인트 |
| :--- | :--- | :--- |
| **`retrieval_hit`** | 기대 근거 문서(`expected["sources"]`)가 검색 결과(`output["sources"]`)에 모두 포함되는가 | 권한상 근거가 없는 케이스(`[]`)는 `score` 없이 `{"label": "해당 없음"}`만 반환 |
| **`citation_present`** | 모델이 실제 답변 텍스트 안에 근거 문서 ID(`[가-힣]+-\d+`)를 인용했는가 | 검색 성공 후 모델의 형식·지시 이행 여부 분리 진단 |
| **`answer_correct`** | HTTP `200` 응답 여부 및 `grade()` 기반 숫자·거절 문구(`"근거를 찾지 못했습니다"`) 일치 여부 | `status != 200`이면 `score` 없이 `{"label": "실행 오류"}`만 반환해 평균 점수 오염 방지 |

### ③ 통제된 실험 비교 & 회귀 테스트 루프
* **Datasets & Experiments:** 동일한 골든 데이터셋(`inputs` / `outputs` / `metadata`) 위에서 프롬프트 등 **단일 변수만 변경해 재실험**함으로써 개선(Improvement)과 퇴행(Regression)을 공정하게 비교합니다.
* **Error Analysis → Regression Asset:** 자동 평가 결과만 믿지 않고 원문·트레이스를 직접 검수(`REVIEW`)한 뒤, 핵심 실패 사례(`Q14`)를 `regression_case.json`으로 박제해 이후 변경 시 재발 여부를 검증합니다.

---

## 3. 실험 관찰 결과 & 병목 분석 (`Qwen2.5-0.5B-Instruct` 기준)

* **지시 이행 누락 (`Q01`, `Q07`, `Q18`):** 검색(`공개-02` 등)과 핵심 숫자 정답(`2일`)은 맞혔으나, 0.5B 모델이 둘째 줄 `근거 : 문서ID` 출력 지시를 누락해 `citation_present = 0.0` 기록.
* **계산·문맥 혼동 + 토큰 절단 (`Q14`, `trace_id: 7cfdffc1e8749e9221ab61aa521910c5`):**
  * 국내 출장(`공개-05`) 질문에 해외 출장(`공개-06`)까지 Top-3에 포함되면서 `42` 대신 `27`을 계산하고, `max_tokens=64` 제한에 걸려 `"1박 20"`에서 문장이 잘림.
  * **워터폴 병목 측정 (`print_tree`):** 총 `50,418.0ms` 중 `retrieval`은 `51.5ms(0.1%)`, `embeddings`는 `4,473.0ms(8.9%)`였으며, **`generation` 단계가 `45,689.2ms(약 90.6%)`로 절대적 병목**을 차지함.
* **권한 차단 후 거절 실패 / 환각 (`Q19`, `Q21`):**
  * RBAC 필터링이 권한 없는 부서(`인사-01`, `재무-01`)를 검색 단계에서 정상 차단했음에도, 모델이 거절하지 않고 임의 추측(`Q19`: `"30%"`)을 하거나 키워드가 겹치는 타 규정(`Q21`: 법인카드 건당 승인 기준 `공개-11`의 `50`)을 월 한도인 것처럼 잘못 인용함.

---

## 4. ⚠️ 조심해야 할 부분 & 흔한 함정 (Pitfalls)

1. **규칙 기반 `grade()` 함수의 치명적 한계**
   * 대괄호(`[...]`)와 한글 문서 ID(`[가-힣]+-\d+`)를 지운 뒤 `re.search`로 **가장 먼저 등장하는 숫자 1개만** 추출합니다.
   * **예시:** 모델이 `"6월 10만원 20만원"`이라고 답하면 단위 구분 없이 앞의 **`6`을 먼저 인식**하며, `"1년 이상 근무 시 15일"`이라고 답해도 **`1`을 추출해 오답 처리**합니다. 반대로 엉뚱한 설명 속에 우연히 기대 숫자만 먼저 나와도 통과되므로 의미·사실성 검증을 대체할 수 없습니다.
2. **부분집합(`issubset`) 기반 검색·인용 평가의 착시**
   * `retrieval_hit`과 `citation_present`는 필수 문서가 포함되었는지만 봅니다. 따라서 `Q14`처럼 오답을 유발하는 불필요한 문서(`공개-06`)가 함께 검색·인용되어도 자동 평가에선 `1.0`으로 통과합니다.
3. **프롬프트 변경 시 버전 메타데이터 동기화 누락 주의**
   * `D69_SYSTEM_PROMPT`만 바꾸고 `D69_PROMPT_VERSION` 환경 변수를 함께 올리지 않으면, Phoenix Experiment 메타데이터와 달리 실제 트레이스 스팬(`gen_ai.prompt.version`)에는 이전 버전(`"v3"`)이 계속 기록됩니다.
4. **E5 임베딩 모델 접두사 규칙 (`multilingual-e5-small`)**
   * 비대칭 검색 모델이므로 문서 벡터화 시 `"passage: "`, 질의 시 `"query: "` 접두사를 빠뜨리면 검색 정확도가 급락합니다.
5. **RBAC 검색은 반드시 Pre-filtering으로 구현**
   * 전체 문서에서 Top-K를 먼저 자른 뒤 권한 필터링을 하면(Post-filtering) 허용 문서가 0개 남을 수 있으므로, 반드시 **권한 필터링 후 Top-K**를 추출해야 합니다.
6. **도커 & 실행 순서 주의사항**
   * **기동 순서:** 반드시 **Phoenix 서버를 먼저 띄우고 `D69_OTLP` 환경 변수를 설정한 뒤** `gen-server` / `rag-server` / `gateway`를 실행해야 OTLP 전송이 활성화됩니다.
   * **Docker 설정:** Phoenix 웹 UI 외부(호스트 브라우저) 접속을 위해 `PHOENIX_HOST="0.0.0.0"` 바인딩이 필요하며, `python:3.12-slim` 이미지에서는 CPU 병렬 연산 라이브러리(`libgomp1`) 설치 및 `HF_HOME` 마운트 캐시 설정이 중요합니다.

---

## 5. ✅ LLM Eval & RAG 구축 체크리스트

### 데이터셋 & 평가 기준
- [ ] 단순 조회 외에 **다중 문서 추론/계산, 권한 차단(거절 유도), 구어체 변형(Paraphrasing)** 케이스를 포함했는가?
- [ ] 평가 대상 앱(`gateway`, `rag-server`, `gen_server`) 코드에 데이터셋의 **참조 답(Ground Truth)이 유출되지 않는가?**
- [ ] 시스템 에러(`status != 200`)와 비적용 항목(`해당 없음`)이 모델 품질 점수 평균을 왜곡하지 않도록 분리했는가?

### 관측성(Observability) & 보안
- [ ] 모든 마이크로서비스가 동일한 `trace_id`로 이어지며, `trace_id`만으로 전체 워터폴(지연·토큰·검색 ID)을 역추적할 수 있는가?
- [ ] 민감한 사용자 질문/답변 원문은 트레이스 태그와 분리하여 별도 접근 제어 저장소에 기록했는가?
- [ ] 사용자 피드백(`/v1/feedback`)이 반드시 유효한 `trace_id`와 매핑되어 저장되는가?

### 실험 비교 & 마무리
- [ ] A/B 비교 시 입력 데이터셋·문서·모델 등 통제 변수를 고정하고 변경 변수만 바꿨는가?
- [ ] 프롬프트/모델 변경 시 버전 환경 변수(`D69_PROMPT_VERSION` 등)도 함께 갱신했는가?
- [ ] 평균 점수만 보지 않고 오답 원문 대조(`REVIEW`) 및 실패 사례의 회귀 테스트(`regression_case.json`) 등록을 마쳤는가?
- [ ] 로컬 실험 종료 후 백그라운드 프로세스(`close_services()`)와 포트를 정리했는가?

---

## 6. 주요 파일 구조

```text
├── d69_otel.py                 # OTel TracerProvider 설정 (JSONL + OTLP/HTTP 이중 내보내기)
├── d69_embeddings.py           # SentenceTransformer 임베딩 로더
├── gen_server.py               # CPU 기반 OpenAI 호환 LLM 생성 서버 (:8690)
├── rag_server.py               # 임베딩 + RBAC 필터링 검색 + LLM 호출 RAG 서버 (:8692)
├── gateway.py                  # API 게이트웨이 및 사용자 피드백 수집 (:8693)
└── work_d69/
    ├── evaluation_cases.jsonl  # 골든 평가 데이터셋 원본
    ├── baseline_responses.json # baseline 실험 응답 결과
    ├── candidate_responses.json# prompt-candidate 실험 응답 결과
    ├── regression_case.json    # 회귀 테스트용 보관 실패 사례 (Q14)
    ├── d69_submission.json     # 최종 판단 및 근거 종합 파일
    ├── content/rag.jsonl       # 질문·답변 원문 로그 (trace_id 연결)
    └── spans/*.jsonl           # 서비스별 OTel 스팬 JSONL 로그
```
