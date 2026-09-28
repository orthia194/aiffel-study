Advanced RAG는 Naive RAG의 단점(낮은 검색 정확도, 노이즈 문서 포함, 환각)을 극복하기 위해 **"검색 전(Pre), 색인 시(Indexing), 검색 후(Post)"** 전 과정에 고도화 로직을 결합한 아키텍처입니다.

---

### 1. Advanced RAG (고급 RAG) 개요

* **정의:** 단순히 질문을 벡터로 바꿔 DB를 뒤지는 Naive RAG에서 벗어나, **질문 재작성·데이터 가공·검색 결과 재정렬** 프로세스를 추가하여 검색 정밀도(Precision)와 소환율(Recall)을 극대화한 구조입니다.
* **핵심 흐름:** `Indexing Refinement` (색인 고도화) ➔ `Pre-retrieval` (질문 재작성/확장) ➔ `Retrieval` (검색) ➔ `Post-retrieval` (후처리 및 압축) ➔ `Generation` (답변 생성)

---

### 2. Indexing Refinement (색인 및 저장 고도화)

문서를 Vector DB에 넣기 전, '어떻게 자르고 어떻게 저장해야 나중에 가장 잘 찾아질까?'를 고민하는 단계입니다.

* **Chunk Size Optimization (청크 크기 최적화):**
* 문맥이 깨지지 않도록 문단/슬라이딩 윈도우(Sliding Window) 단위로 적절한 Overlap(중복) 구간을 두어 청킹합니다.


* **Parent-Document Retriever (부모-자식 문서 기법):**
* **자식 청크(작은 조각):** 검색 정밀도를 높이기 위해 임베딩 및 검색용으로 사용합니다.
* **부모 청크(큰 문맥):** 실제 LLM에 전달할 때는 자식 청크가 속한 큰 전체 문맥(Parent)을 가져와 맥락 손실을 막습니다.


* **Hierarchical Indexing (계층적 색인):**
* 문서 요약본(Summary)을 먼저 검색한 뒤, 관련 있는 세부 청크로 들어가는 2단계 색인 구조를 활용합니다.


* **Metadata Attachment (메타데이터 부가):**
* 날짜, 카테고리, 작성자, 문서 출처 등의 메타데이터를 함께 저장하여 검색 시 필터링(Filtering)을 가능하게 합니다.



---

### 3. Pre-retrieval Process (검색 전 처리)

유저의 질문(Query)을 Vector DB에 던지기 전, **질문의 질을 높여서 정답 탐색률을 극대화**하는 단계입니다.

* **Multi-Query (다중 쿼리):**
* 유저의 한 마디 질문을 LLM이 의도가 같은 여러 형태의 질문으로 확장(Re-write)하여 수평적으로 그물을 널리 칩니다.


* **HyDE (Hypothetical Document Embeddings):**
* LLM이 상식만으로 '가상의 정답 문서'를 먼저 작성해 본 뒤, 그 가상 문서의 어조/형태를 이용해 Vector DB에서 유사한 진짜 문서를 찾아냅니다.


* **Query Decomposition / Step-Back Prompting:**
* 복잡하거나 추상적인 질문을 여러 개의 세부 원자 질문(Sub-queries)으로 쪼개어 단계별로 검색을 수행합니다.



---

### 4. Post-retrieval Process (검색 후 처리)

Vector DB에서 상위 $K$개(Top-K) 문서를 가져온 뒤, **LLM에게 전달하기 직전에 데이터를 요약·정리**하는 단계입니다.

* **Reranking (리랭킹):**
* 벡터 유사도만으로는 한계가 있으므로, Cross-Encoder 같은 정밀 재정렬 모델을 사용해 문서와 질문 간의 실제 관련성을 재평가하여 최상위 순위를 다시 매깁니다.


* **Context Compression (문맥 압축):**
* 가져온 문서 조각에서 질문과 상관없는 쓰레기 텍스트(Noise)를 제거하고 정답과 직결된 핵심 문장만 압축하여 프롬프트에 넣습니다.


* **Lost in the Middle 방지 (순서 배치):**
* LLM은 컨텍스트의 맨 앞과 맨 뒤 내용을 가장 잘 기억합니다. 가장 중요한 정답 문서를 프롬프트의 맨 앞이나 맨 뒤에 배치하도록 재정렬합니다.



---

### 💡 한 눈에 보는 요약 표

| 분류 | 주요 기술 | 핵심 목적 |
| --- | --- | --- |
| **Indexing Refinement** | Parent-Document, Hierarchical Indexing | 문맥 손실 방지 및 고품질 데이터베이스 구축 |
| **Pre-retrieval** | Multi-Query, HyDE, Query Decomposition | 유저의 애매한 질문을 정제하고 탐색 범위 확장 |
| **Post-retrieval** | Reranking, Context Compression | 노이즈 제거, 순위 재정렬을 통해 LLM 환각 최소화 |
