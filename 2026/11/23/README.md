# 🏠 가정용 8GB VRAM 로컬 환경 기반 규약집 RAG + 경량 LLM (4B AWQ)


>(총 4일 진행) **부부/가구 공동생활 규약집(V1 / V2 / V3)**을 기반으로 **다중 판본·경과조치 판별, 잡담 폭탄(Haystack) 방어, 출처 조항/표/쪽 기반 근거 판정**을 수행하는 경량 로컬 RAG 파이프라인 및 Plan B 지식증류 4B 모델 프로젝트입니다.

---

## 📌 핵심 목표 및 하드웨어 스펙

- **목표 환경**: 가정용 **8GB VRAM PC**에서 단독 로컬 구동 가능 (LLM + Embedding + Reranker 동시 적재)
- **LLM (Plan B)**: `Qwen3-32B` → `Qwen3-4B-Instruct-2507` 직접 지식증류 + 비대칭 4-bit AWQ (`W4A16_ASYM`, **2.48GB**)
- **Embedding**: `nlpai-lab/KURE-v3` (8-bit 양자화, ~2GB VRAM)
- **Reranker**: `nlpai-lab/KURE-Reranker-nano` (`max_length=8192`)
- **Vector DB**: `Qdrant` (로컬 파일 기반 `./qdrant_db`)

---

## 🏗 전체 시스템 아키텍처 & 파이프라인

### 1. 데이터 전처리 & 실시간 자동 동기화 파이프라인
```text
[./pdf_dir/*.pdf] (신규·수정 PDF 투입, SHA-256 해시 감시)
        │
        ▼ (pdf_processor_final.py : PyMuPDF + Kiwi)
  ├─ 1. 본문 영역 클리핑 (y: 60~785) 및 표(bbox) 영역 우선 탐지 → 마크다운 변환
  ├─ 2. 본문-표 좌표 중복 제거 및 (y0, x0) 읽기 순서 복원
  ├─ 3. 폰트 크기·Bold 비율 기반 계층 메타데이터 태깅 (chapter / topic / article / table)
  ├─ 4. Kiwi 형태소 분석기(glue) + 도메인 규칙으로 한국어 줄바꿈·표 <br> 문장 복원
  └─ 5. 페이지 단위 청킹 & 원자적 교체(.tmp → os.replace)
        │
        ├─▶ [./cleaned_md/*.md] (육안 검수용 마크다운)
        └─▶ [./chunks_dir/*.jsonl]
                │
                ▼ (rag_retriever_final_test2.py : Watch / Sync)
          ├─ 파일 해시·mtime 스냅샷 감지 → 증분 업서트 (UUIDv5 결정적 ID)
          ├─ 파일명/메타/본문에서 판본 자동 추출 + 최고 버전 자동 is_latest=True 지정
          ├─ 표 01(문서 관리)에서 판본별 시행 기간(effective_from ~ to) 자동 파싱
          └─ [./qdrant_db] (KURE-v3 임베딩 적재)
```

### 2. 실시간 질의응답(RAG + LLM) 추론 파이프라인
```text
[사용자 질의] (최대 6,000자+ 잡담 폭탄 / 전각 숫자 / 이모지 / 이형 날짜 포함)
        │
        ▼ 1. 질의 정제 & 잡담 폭탄 방어 (clean_query_text + extract_core_query)
  ├─ NFKC 전각→반각 정규화, 자모 스팸·이모지·상담 메모 래퍼 제거 (줄바꿈 보존)
  └─ 450자·6문단 초과 시: 문단 배치 임베딩(KURE-v3) → Qdrant 매칭 → Reranker 채점
     → 잡담 중앙값 대비 관련 문단 + 짧은 핵심 신호(75자 이하 ?/숫자/선택지)만 압축
        │
        ▼ [핵심 질의 core_q] (예: 6,243자 → 74자 압축)
        │
        ├─▶ 2. 판본/날짜 동적 라우팅 (detect_requested_version + analyze_query_dates)
        │     ├─ 명시적 판본(V1/V2/V3) 또는 비교(ALL) → 해당 판본 검색
        │     ├─ 과거 날짜(V1: 6~9월 / V2: 10~11월) + 개정 항목(표 07)
        │     │    → 통합 합본 부칙(page_53~57) 최상단 고정 + 과거 본문 조합 (중복 chunk_id 제거)
        │     └─ 일반 질의 → 최신 판본(V3) 단독 검색
        │
        ├─▶ 3. 2단계 검색 + 안전 Fallback
        │     └─ KURE-v3 (Top-15~20) → KURE-Reranker-nano (min_score=0.0, 미달 시 Top-1 >= -1.8 Fallback)
        │
        └─▶ 4. 동적 프롬프트 & 로컬 LLM 추론 (vLLM : verdict-4b)
              ├─ 기본: [핵심 계산·조건 확인] 3~5줄 → [최종 답변] → [참고 근거]
              ├─ 과거 날짜·개정 항목 시: 기준일 및 부칙 적용 힌트(date_hint) 주입
              └─ 300~600자 분량 요청 감지 시: '3~5줄 간결' 제약 동적 해제 (build_rule_4)
```

### 3. Plan B: 경량 4B LLM 지식증류 & 비대칭 양자화 파이프라인 (Colab G4)
```text
[Teacher: Qwen/Qwen3-32B (Thinking, temp=0.6)]
   ├─ 범용 한국어 상황 지문 300개 + 문맥 반전·함정 질의 생성
   ├─ 4단계 검증 (JSON → 원문 인용 → 3-Pass 만장일치 → Auditor NLI 검증)
   └─ '판단 불가' 63건(1:1:1 비율) 보강 → gold_distill_train.jsonl (총 964건, Val 47건)
        │
        ▼ Offline Top-64 Logit 캐싱 (0.5*CE + 0.5*Top-K KL, T=2.0)
[Student: Qwen/Qwen3-4B-Instruct-2507 (Non-thinking)] + LoRA(r=32, alpha=64, 5 epochs)
        │
        ▼ LoRA 병합 (final_student_4b_bf16, Val 85.11%)
[llmcompressor 비대칭 AWQ]
   ├─ AWQModifier(duo_scaling="both") + QuantizationModifier(scheme="W4A16_ASYM", ignore=["lm_head"])
   └─ 512샘플 보정 → final_student_4b_awq_asym (2.48GB, Val 87.23% 방어)
```
> *참고: 실제 규약집 원문(`chunks_dir/*.jsonl`)은 LLM 학습에 일절 사용하지 않은 Zero-shot Domain Adaptation 상태이며, `Qwen3-4B-Instruct-2507`은 응답 속도와 형식 지시 이행을 위해 프루닝 없이 Non-thinking 모델을 기반으로 직접 증류했습니다.*

---

## 📁 주요 파일 및 디렉터리 설명

### 핵심 코드 모듈
| 파일명 | 역할 및 주요 기능 |
|---|---|
| **`pdf_processor_final.py`** | PDF 자동 전처리 및 페이지 단위 청킹 모듈.<br>- `PyMuPDF` 표/본문 분리, 폰트 크기·Bold 기반 메타데이터 태깅, `Kiwi` 줄바꿈 복원<br>- `.pdf_sync_state.json` SHA-256 해시 비교로 증분 변환 및 `.tmp` 원자적 교체 (`--watch` 지원) |
| **`rag_retriever_final_test2.py`** | **(최신 RAG 모듈)** 실시간 버전 관리형 규약집 검색기.<br>- `clean_query_text()`, `extract_core_query()`: 전각 정규화 + 6,000자 잡담 폭탄 압축<br>- `sync_folder()`, `upsert_version()`: `chunks_dir` 자동 동기화 및 최고 판본 자동 최신 지정<br>- `analyze_query_dates()`, `search_with_date_routing()`: 질문 내 날짜 추출(분수/`N개월` 오탐 차단), `표 07` 부칙 페이지(`page_53~57`) 최상단 고정, `chunk_id` 중복 차단<br>- `search()`: `KURE-v3` + `KURE-Reranker-nano` + `-1.8` 안전 Fallback (`rag_retriever_final.py` / `test1.py`의 개선 최종판) |
| **`4b_awq_eval_100_public12.py`** | **(최신 100문항 평가 스크립트)** `NewDataSet_V3` 100문항 평가.<br>- `core_q` 기준 명시적 판본 판별 (`detect_requested_version` 일상 단어 오탐 방지)<br>- 날짜 기반 과거 판본 라우팅 및 `date_hint` 동적 주입<br>- `300~600자` 장문 요청 감지 시 `build_rule_4()` 동적 프롬프트 전환 (`public6.py` 대비 날짜/장문 약점 보완) |
| **`make_review.py`** | 평가 결과(`.jsonl`)에서 잡담 원문을 압축하고 `search_mode`, `detected_dates`, `sources`, `GT`, `PRED`를 가독성 있게 추출(`review_1_50.txt`, `review_51_100.txt`)하는 검수 유틸리티. |

### 주요 디렉터리 구조
```text
/workspace/dlthon2/
├── pdf_dir/                     # 원본 규약집 PDF (data.pdf, data_V3.pdf 등)
├── chunks_dir/                  # 전처리된 페이지 단위 청크 (data.jsonl, data_V3.jsonl)
├── cleaned_md/                  # 전처리 검수용 마크다운 파일
├── qdrant_db/                   # 로컬 Qdrant 벡터 DB 저장소
├── final_student_4b_awq_asym/   # 최종 양자화 모델 (W4A16_ASYM, 2.48GB)
├── NewDataSet/                  # [평가 전용] V2 기준 100문항 (학습·RAG 투입 절대 금지)
└── NewDataSet_V3/               # [평가 전용] V3 기준 100문항 (학습·RAG 투입 절대 금지)
```

---

## 📊 성능 변천 및 실험 결과

### 1. 양자화 방식별 검증 성능 비교 (`Val` 47건 기준)
| 모델 / 양자화 방식 | 용량 | Val 정확도 | 비고 |
|---|---:|---:|---|
| `final_student_4b_bf16` (증류 병합 원본) | ~8.0GB | 85.11% | 8GB VRAM 단독 구동 불가 |
| GPTQ (4-bit) | ~2.5GB | 76.60% | 양자화 손실 큼 (-8.51%p) |
| 대칭 AWQ (4-bit) | ~2.5GB | 80.85% | 양자화 손실 (-4.26%p) |
| **비대칭 AWQ (`W4A16_ASYM`, `duo_scaling="both"`)** | **2.48GB** | **87.23%** | **BF16 이상 성능 방어 & 8GB 스펙 충족** |

### 2. RAG 파이프라인 고도화별 100문항 평가 추이
- **V2 (`NewDataSet`, 100문항)**: `58/100` (초기) → **`70/100`** (잡담 압축 `extract_core_query` 적용, **+12점**) → `74/100` (프롬프트·리랭커 미세조정)
- **V3 (`NewDataSet_V3`, 100문항)**:
  - `public6.py` (기본 V3): **79 / 100 (79.0%)** *(Part1 18/25 | Part2 24/25 | Part3 18/25 | Part4 19/25)*
  - `public12.py` (`test2` 통합): 장문(`300~600자`) 10문항 상세 서술 복구, 무한 루프 0건, 날짜 기반 V2/V3 분기(`PUB-054` 5거래일·`PUB-091` 13만 원 등) 개선 및 **RAG 검색 재현율 95%+** 달성

---

## 🚀 실행 및 평가 가이드

### 1. vLLM 모델 서빙
```bash
vllm serve ./final_student_4b_awq_asym \
  --served-model-name verdict-4b \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  --enforce-eager
```

### 2. PDF 전처리 감시 (선택)
```bash
python pdf_processor_final.py --pdf-dir ./pdf_dir --chunks-dir ./chunks_dir --md-dir ./cleaned_md --watch
```

### 3. 100문항 평가 및 리뷰 추출
```bash
python 4b_awq_eval_100_public12.py
python make_review.py
```

---

## ⚠️ 운영 및 개발 시 주의사항

1. **추론 파라미터 고정 (`frequency_penalty=0.0`)**
   - `frequency_penalty > 0` 설정 시 계산 과정에 등장한 정답 키워드(`16시`, `감석` 등)를 최종 결론에서 회피하는 부작용이 발생하므로 반드시 `0.0`을 유지합니다. (`repetition_penalty=1.04` 역시 일부 문항 성능 저하 이력이 있어 미적용 상태 유지)
2. **평가 전용 데이터셋 격리**
   - `NewDataSet/`과 `NewDataSet_V3/`는 **절대 학습 데이터나 RAG 검색 DB(`chunks_dir`)에 투입하지 않습니다.**
3. **macOS NFD 한글 파일명 대응**
   - 평가 파일명이 NFD 자모 분리 형태로 저장될 수 있으므로, 파일 경로 지정 시 하드코딩 대신 `glob.glob("./NewDataSet_V3/*01*.jsonl")` 패턴을 사용합니다.

---

## 🔧 현재 한계 및 팀 인계 (향후 개선 포인트)

- **현재 강점**: RAG 파이프라인(`KURE-v3` + `Qdrant` + `KURE-Reranker-nano`)의 **근거 검색 재현율은 95% 이상**이며, 6,000자 잡담 방어 및 신규 판본(V4+) 무중단 자동 반영 구조가 완성되어 있습니다.
- **남은 약점 (4B 모델 자체 추론 한계)**:
  1. 다단계 산수·부등호·시각 대소 비교 실수 (`PUB-024`, `025`, `033`, `044`, `045`, `054`, `062`, `095`)
  2. 핵심 값 누락 시 `추가 확인` 대신 임의 단정 (`PUB-003`, `029`, `079`, `089`, `099`)
  3. 예외 우선순위·표 행/열·V1 부칙 수치 혼동 (`PUB-006`, `027`, `057`, `069`, `094`)
- **권장 후속 작업**:
  - **서비스 통합**: 검증된 `rag_retriever_final_test2.py` + `4b_awq_eval_100_public12.py`의 라우팅/프롬프트 로직을 최종 FastAPI 서빙(`main_rag.py`)에 이식
  - **도메인 특화 도약 (85~90점+ 목표 시)**: 평가셋을 배제한 `./chunks_dir` 원문 기반으로 32B 규약집 특화 증류 데이터 생성 및 4B 추가 학습 + `W4A16_ASYM` 재양자화 (또는 상위 모델 교체 테스트)
