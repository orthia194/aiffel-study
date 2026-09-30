# 🚀 RAG Pipeline & Evaluation Workflow

본 프로젝트는 데이터 전처리부터 **RAGAS(RAG Assessment)** 기반의 품질 평가 및 서비스 배포까지의 End-to-End 엔지니어링 프로세스를 준수하여 구축되었습니다.




[1. Chunking] ➔ [2. Embedding] ➔ [3. Vector DB] ➔ [4. Hybrid Retrieval]
➔ [5. Reranking] ➔ [6. Prompting] ➔ [7. Generation] ➔ [8. RAGAS Eval]


---

## 📌 Phase 1. Data Ingestion & Indexing (데이터 구축 단계)

### 1. Document Chunking (문서 분할)

* **설명:** 대용량 문서(PDF, Markdown, Web Text 등)를 LLM의 컨텍스트 윈도우 한계와 검색 정밀도를 고려해 적절한 크기로 자릅니다.
* **주요 기법:** `RecursiveCharacterTextSplitter` 활용 (예: Chunk Size = 500, Chunk Overlap = 50)
- **Recursive Character Chunking:** 문맥 끊김 방지를 위해 `Chunk Size: 500`, `Overlap: 50` 기반 표준 분할 적용.
- **Markdown / Document Structure Splitter:** 문서 내 헤더(`#`, `##`) 및 표(Table) 구조를 보존하여 검색 정확도 향상.
- **Semantic Chunking (고급):** 임베딩 유사도 기반의 주제 전환점 감지를 통해 글자 수가 아닌 의미 단위로 청크 분할.
- **Parent-Document Retrieval:** 검색용 Small Chunk(200자)와 LLM 전달용 Parent Chunk(1000자)를 분리하여 context_precision 및 recall 동시 최적화.

### 2. Embedding (벡터화)

* **설명:** 분할된 텍스트 청크(Chunk)를 고차원 수치 벡터(Dense Vector)로 변환합니다.
* **사용 모델:** `text-embedding-3-small` / OpenAI / HuggingFace HuggingFace models

### 3. Vector DB Indexing (지식 저장소 구축)

* **설명:** 생성된 벡터와 메타데이터(문서 출처, 페이지 번호 등)를 빠른 유사도 검색이 가능한 전용 DB에 저장합니다.
* **주요 솔루션:** ChromaDB, Pinecone, FAISS, Qdrant 등

---

## 🔍 Phase 2. Retrieval & Generation (검색 및 생성 단계)

### 4. Hybrid Retrieval (검색 파이프라인)

* **설명:** 사용자 질의(Query)가 입력되면 의미 기반 검색(Dense Retrieval)과 키워드 기반 검색(Sparse Retrieval, 예: BM25)을 결합하여 지식을 수집합니다.
* **기술:** Reciprocal Rank Fusion (RRF)을 통해 두 검색 결과의 순위를 통합.

### 5. Reranking & Filtering (검색 결과 재정렬)

* **설명:** 1차 검색된 Top-N 문서 중 질문과의 연관성이 높은 순으로 Cross-Encoder 모델을 이용해 다시 정렬하고, 불필요한 노이즈 문단을 제거합니다.

### 6. Prompt Augmentation (프롬프트 합성)

* **설명:** 정렬된 최적의 컨텍스트(Contexts)와 사용자 질문, 시스템 페르소나 지시문을 하나로 합성하여 LLM 전달용 프롬프트를 만듭니다.

### 7. LLM Generation (근거 기반 답변 생성)

* **설명:** LLM이 전달받은 컨텍스트만을 참고하여 지어내는 내용(Hallucination) 없이 정밀한 답변을 출력합니다.

---

## 📊 Phase 3. Quality Evaluation & Monitoring (RAGAS 평가 및 개선)

### 8. RAGAS Evaluation (자동 품질 검증)

* **설명:** `LLM-as-a-Judge` 메커니즘을 적용하여 4대 핵심 지표로 파이프라인의 성능을 수치화합니다.
* **합성 데이터 생성:** RAGAS의 `TestsetGenerator`를 통해 사내 문서로부터 200~500개의 평가용 질문/정답을 자동 생성하여 오프라인 평가 수행.

| 영역 | 지표 (Metric) | 설명 |
| --- | --- | --- |
| **Retrieval** | `context_precision` | 검색된 문서 중 필요한 핵심 정보가 상위에 위치하는가 |
| **Retrieval** | `context_recall` | 모범 답안을 작성하는 데 필요한 지식이 빠짐없이 검색되었는가 |
| **Generation** | `faithfulness` | 답변이 검색된 컨텍스트에만 근거하여 작성되었는가 (환각 검증) |
| **Generation** | `answer_relevancy` | 답변이 사용자 질문의 의도에 부합하는가 |

---

## 🛠 RAGAS Evaluation Quickstart

```python
import nest_asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

nest_asyncio.apply()

# 1. Dataset 준비
dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
})

# 2. Judge LLM & Embeddings 설정
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. 평가 실행
results = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

# 4. 리포트 출력
print(results)
df = results.to_pandas()

```
---

### 1. 문맥 유지 및 구조적 청킹 기법

#### ① 문자/구분자 기반 청킹 (Character / Recursive Character Chunking)

가장 기본적이며 실무에서 80% 이상 기본값으로 사용하는 방식입니다.

* **작동 방식:** 단락 구분 기법(`\n\n`), 줄바꿈(`\n`), 띄어쓰기(` `) 순서로 구분자(Delimiter)를 우선순위에 따라 적용하여 설정한 **Chunk Size**(예: 500자)와 **Chunk Overlap**(예: 50자) 크기에 맞춰 텍스트를 나눕니다.
* **장점:** 속도가 매우 빠르고 구현이 간결하며, `Overlap` 영역 덕분에 문장이 잘리는 경계면의 문맥 손실을 줄일 수 있습니다.
* **단점:** 문서의 의미적 단락이나 표(Table), 코드 블록 등의 논리적 구조를 무시하고 잘릴 수 있습니다.

#### ② 문서 구조 기반 청킹 (Markdown / HTML / Code Splitter)

문서의 헤더(`#`, `##`)나 태그, 코드 구문을 인지하여 나누는 방식입니다.

* **작동 방식:** 마크다운의 `# Header`, HTML의 `

`, `

`태그, 또는 파이썬의`def` 함수 단위 등 문서의 **포맷 구조를 파악해 독립된 섹션 단위**로 잘라냅니다.

* **장점:** 표나 소제목에 속한 내용이 하나의 청크로 유지되어, 질문에 답할 때 관련 맥락 전체를 온전히 가져올 수 있습니다.
* **적용 문서:** API 문서, 기술 블록, 위키 페이지, 보고서.

---

### 2. 고성능 / AI 기반 고급 청킹 기법

#### ③ 시맨틱 청킹 (Semantic Chunking)

글자 수가 아닌 **문장 간의 의미(Semantic) 변화**를 기준으로 나눕니다.

* **작동 방식:** 문장 단위로 쪼갠 뒤 임베딩 모델을 통해 각 문장 간의 유사도를 연속으로 측정합니다. 이야기나 주제가 전환되어 **문장 간 유사도가 갑자기 떨어지는 지점(Threshold)**을 찾아 청크를 나눕니다.
* **장점:** 하나의 청크 안에는 오직 '동일한 주제의 이야기'만 담기므로 RAG의 환각을 크게 낮춥니다.
* **단점:** 임베딩 모델을 계속 호출해야 하므로 전처리 시간이 오래 걸리고 비용이 발생합니다.

#### ④ 에이전틱 / 계층적 청킹 (Agentic / Hierarchical Chunking)

LLM을 활용하거나 **소형 청크 + 대형 청크** 구조를 조합하는 고도화 기법입니다.

* **작동 방식 (Parent-Document):** 검색은 짧고 명확한 **소형 청크(Small Chunk)**로 수행하여 정확도를 높이고, LLM 답변 생성 시에는 그 소형 청크가 포함된 **대형 부모 청크(Parent Chunk/전체 단락)**를 넘겨주는 방식입니다.
* **작동 방식 (Agentic):** LLM에게 *"이 문서를 읽고 의미 단위로 가장 완벽한 위치에서 청크를 나누어 줘"*라고 요청합니다.
* **장점:** 검색 정밀도와 답변 생성에 필요한 문맥(Context) 크기를 동시에 최적화할 수 있습니다.

#### ⑤ 도메인 특화 청킹 (Domain-Specific Chunking)

일반 텍스트가 아닌 특수 포맷 문서에 맞춤 적용하는 방식입니다.

* **작동 방식:**
* **표/차트:** Vision LLM(OCR)을 사용해 표를 Markdown이나 JSON 형태로 복원한 뒤 표 1개를 하나의 청크로 묶음.
* **법률/규정:** '제N조 N항' 단위로 자르는 정규식 패턴 기반 청킹.
