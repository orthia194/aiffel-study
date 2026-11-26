# Apache Airflow 3.3.2 학습 정리 & LLMOps RAG 파이프라인 실습

> **실습 환경:** Docker · Python 3.12 · JupyterLab · Apache Airflow 3.3.2  
> **실습 주제:** 사내 규정 문서 수집·정제·청킹·인덱싱·회귀 평가·배포 승격을 오케스트레이션하는 LLMOps 파이프라인 구축 및 장애/복구 시뮬레이션

---

## 1. Apache Airflow란?

**Apache Airflow**는 복잡한 작업 흐름(워크플로우)을 **Python 코드로 작성(Author), 스케줄링(Schedule), 모니터링(Monitor)**하는 오픈소스 워크플로우 오케스트레이션 플랫폼입니다.

* **핵심 비유 — "오케스트라의 지휘자":** Airflow는 Spark나 LLM처럼 데이터를 직접 대량 가공·연산하는 엔진이 아니라, 각 단계의 작업 순서, 타이밍, 의존성, 재시도, 실패 처리를 조율하는 지휘자 역할을 합니다.
* **주요 장점:**
  * **Workflow as Code:** Python 코드로 파이프라인을 정의하므로 Git 버전 관리, 코드 리뷰, 모듈화, 테스트가 가능합니다.
  * **강력한 스케줄링 & 재처리:** 작업 간 선후 관계를 보장하고, 실패 시 자동 재시도(Retry)와 과거 데이터 재처리(Backfill)를 지원합니다.
  * **직관적인 웹 UI:** Graph, Grid, Logs, XCom 등을 통해 실행 상태와 원인을 한눈에 추적하고 특정 지점부터 재실행할 수 있습니다.

---

## 2. Airflow 핵심 개념 (Core Concepts)

| 개념 | 설명 |
| :--- | :--- |
| **DAG** | *Directed Acyclic Graph*(방향성 비순환 그래프). 전체 파이프라인 설계도로, 한 방향으로만 흐르며 순환(무한 루프)을 허용하지 않습니다. |
| **DAG Run** | 정의된 DAG가 특정 시각이나 수동 트리거에 의해 **실제로 한 번 실행된 인스턴스**입니다. |
| **Task / Task Instance** | DAG를 구성하는 개별 작업 단위이며, 특정 DAG Run 안에서 실행 중인/실행된 개별 태스크가 Task Instance입니다. |
| **Operator / Sensor / Hook** | Task가 수행할 작업 템플릿(`BashOperator`, `PythonOperator` 등), 특정 조건을 대기하는 `Sensor`, 외부 시스템 연결 인터페이스 `Hook`으로 구성됩니다. |
| **TaskFlow API (`@dag`, `@task`)** | 파이썬 함수 데코레이터로 직관적인 DAG를 작성하는 방식입니다. Airflow 3에서는 **`airflow.sdk`** 네임스페이스를 사용합니다. |
| **XCom** | *Cross-Communication*. 서로 독립된 프로세스/워커에서 실행되는 Task 간에 작은 메타데이터를 주고받는 저장소입니다. |
| **Asset** | 시간 기반 스케줄링뿐만 아니라 특정 데이터 산출물(Asset)이 갱신될 때 후속 DAG를 트리거하는 **데이터 인지(Data-Aware) 스케줄링** 개념입니다. |

### 💡 TaskFlow API의 의존성 연결 원리
`@task`로 데코레이트된 함수를 호출하면 실제 결괏값이 즉시 계산되는 대신, 미래의 XCom 출력을 가리키는 **`XComArg` 참조 객체**가 반환됩니다. 이 반환값을 다음 태스크 함수의 인자로 넘기면 Airflow가 **"후속 태스크가 선행 태스크의 출력을 입력으로 사용한다"**고 인식해 **데이터 전달(XCom)과 실행 순서(의존성)를 동시에 연결**합니다.

### 💡 XCom에 데이터 본문 대신 "파일 경로"만 전달하는 이유
XCom 값은 Airflow **메타데이터 DB**에 저장됩니다. 수 MB~GB의 원문 문서, 청크, 인덱스 데이터를 직접 XCom에 넣으면 DB 용량 폭증, 직렬화(JSON) 부하, 성능 저하 및 크기 제한 오류가 발생할 수 있습니다. 따라서 무거운 실제 데이터는 스토리지(로컬 디스크, S3 등)에 저장하고 XCom에는 가벼운 **파일 경로 문자열만 전달**합니다.

---

## 3. Airflow 3.x 아키텍처 & 주요 특징

### 컴포넌트 역할
| 컴포넌트 | 역할 |
| :--- | :--- |
| **Scheduler** | DAG의 실행 조건(시간/의존성)을 감시하고 실행할 Task를 Executor에게 전달합니다. |
| **Executor & Worker** | Task를 어디서 어떻게 실행할지 결정하고(Executor) 실제 코드를 수행합니다(Worker). |
| **DAG Processor** | `dags/` 폴더의 Python 파일을 주기적으로 파싱하여 메타데이터 DB에 등록합니다(Scheduler와 분리). |
| **API Server & UI** | FastAPI 기반 REST API(`/api/v2`) 및 React Flow 기반 웹 UI를 제공하며, Worker의 실행 통신 창구 역할도 맡습니다. |
| **Metadata DB** | DAG 구조, 실행 이력, Task 상태, XCom, Asset 이벤트 등을 저장합니다(실습에서는 SQLite 사용). |
| **Triggerer** | *(선택)* 비동기(Deferrable) Sensor나 장시간 대기 작업을 `asyncio` 기반으로 가볍게 감시합니다. |

### 실습에서 확인한 Airflow 3.x 핵심 포인트
* **Task SDK (`airflow.sdk`):** DAG/Task 작성 API가 `airflow.sdk`(`Param`, `dag`, `task`, `AirflowFailException`, `AirflowSkipException`)로 분리되었습니다.
* **Task Execution API (`/execution/`):** Worker가 메타데이터 DB에 직접 붙지 않고 API Server를 거쳐 통신합니다. 따라서 웹 포트를 `8680`으로 변경할 때 `AIRFLOW__CORE__EXECUTION_API_SERVER_URL=http://127.0.0.1:8680/execution/`도 함께 맞춰주어야 합니다.
* **설정의 환경변수 분리:** `[api]`(`AIRFLOW__API__PORT`), `[dag_processor]`(`REFRESH_INTERVAL` 등), `SimpleAuthManager`(`SIMPLE_AUTH_MANAGER_ALL_ADMINS`로 로컬 실습용 자동 인증) 등 역할별 설정 구조가 명확해졌습니다.
* **DAG Versioning & DB 마이그레이션:** UI에서 `Dag 버전 v1`처럼 버전 추적이 가능하며, DB 초기화 명령어는 `airflow db migrate`로 통일되었습니다.

---

## 4. 실습 환경 및 프로젝트 구조

### Docker + JupyterLab + Airflow 3.3.2 구성
* **이미지 확장 (`Dockerfile`):** `apache/airflow:3.3.2-python3.12` 기반에 `--constraint` 공식 제약 파일을 함께 지정해 라이브러리 버전 충돌 없이 `jupyterlab`을 설치했습니다.
* **컨테이너 설정 (`compose.yaml`):**
  * `user: "${D68_UID:-50000}:0"` 매핑으로 호스트와 컨테이너 간 파일 소유권 충돌 방지
  * 포트 포워딩: `8888`(JupyterLab), `8680`(Airflow UI)
  * 상위 실습 폴더를 `/workspace`로 Bind Mount하여 컨테이너 종료 후에도 결과물 보존
* **프로세스 그룹 관리:** `airflow standalone`은 `api-server`, `scheduler`, `dag-processor`, `triggerer`를 하위 프로세스로 함께 띄우므로 `start_new_session=True`로 새 세션을 만들고 종료 시 `os.killpg(..., SIGTERM)`로 프로세스 그룹 전체를 정리합니다.

### 파일 구성
```text
work_d68/
├── airflow/                       # AIRFLOW_HOME (airflow.db, 설정 및 로그)
├── dags/
│   ├── d68_pipeline.py            # [로직 1] 문서 스냅샷·필터링·중복제거·청킹·드리프트 감지
│   ├── d68_release.py             # [로직 2] 버전별 인덱싱·회귀평가(flip)·승격·롤백 포인터 관리
│   └── d68_weekly_refresh.py      # [DAG] 운영 DAG(d68_weekly_refresh) & 리허설 DAG(d68_ops_rehearsal)
├── corp_docs/                     # 가상의 사내 공유 드라이브 문서
├── blobs/                         # SHA-256 해시 기반 원문 저장소 (중복 없는 내용 보관)
├── snapshots/                     # 일자별 문서 목록 스냅샷 (YYYY-MM-DD.json)
├── parts/                         # 운영/사전점검 일자별 청크 및 평가 보고서
├── ops_rehearsal/                 # UI 리허설용 격리 출력 폴더
├── indexes/                       # 버전별 불변 검색 인덱스 (idx-<해시>/index.json)
├── releases/                      # 출시 번들 JSON 및 current.json (실제 서빙 포인터)
├── run_one.py                     # dag_obj.test() 기반 격리된 E2E 테스트 스크립트
└── d68_submission.json            # 최종 실습 결과 및 판단 증적 JSON
```

---

## 5. 파이프라인에 적용된 실무 엔지니어링 패턴

### ① 관심사 분리 (Airflow 독립적인 비즈니스 로직)
`d68_pipeline.py`와 `d68_release.py`는 **Airflow를 전혀 `import`하지 않는 순수 Python 모듈**로 작성했습니다.
* Airflow를 띄우지 않고도 빠른 단위 테스트와 재사용이 가능합니다.
* Airflow의 `DAG_DISCOVERY_SAFE_MODE`에 의해 `airflow`와 `DAG` 키워드가 모두 없는 파일은 DAG 파싱 대상에서 안전하게 제외됩니다.

### ② 재현성과 불변성 (Snapshot + Content-Addressable Storage)
* **스냅샷 & Blob:** `stage_catalog()`는 문서 내용의 SHA-256 해시를 계산해 `blobs/<sha>`에 한 벌만 저장하고, 그날의 `(경로, 해시, 크기)` 목록을 `snapshots/YYYY-MM-DD.json`으로 박제합니다. 같은 날짜를 재실행하거나 백필할 때 현재 드라이브 상태가 섞이는 것을 막아 **완벽한 재현성**을 보장합니다.
* **불변 인덱스 & 원자적 쓰기:** `(스냅샷 해시 + 청킹 설정 + 검색 방식)` 조합 해시로 `indexes/idx-<해시>/` 폴더를 따로 만들어 서비스 중인 인덱스를 건드리지 않으며, `.tmp` 임시 파일에 다 쓴 뒤 `replace()`하는 `write_atomic()`으로 반쯤 쓰인 파일 생성을 차단합니다.

### ③ 타임존 변환과 스케줄링
* DAG의 `start_date`에 `pendulum.datetime(2026, 9, 1, tz="Asia/Seoul")`을 지정해 크론 표현식 `"0 3 * * 1"`이 **한국 시각 매주 월요일 03:00**에 동작하도록 설정했습니다.
* Airflow 내부 DB는 `logical_date`를 UTC로 다루므로 `day_of()`에서 `astimezone(ZoneInfo("Asia/Seoul"))`로 변환해 정확한 한국 날짜 파티션(`YYYY-MM-DD`)을 계산합니다.

### ④ 다단계 품질 게이트 & 회귀 평가 (`flips`)
1. **하드 게이트 (`prepare`):** 필터링 통과율이 30%(`min_rate=0.3`) 미만이면 후속 처리를 차단합니다.
2. **소프트 게이트 (`soft_chunk_drift`):** 직전 파티션 대비 청크 수가 ±15% 이상 급변하면 파이프라인을 멈추진 않되 `warnings` 경고를 남깁니다.
3. **챔피언 vs 챌린저 회귀 평가 (`evaluate` · `decide_promotion`):** 새 인덱스(`cand`)와 현재 서비스 중인 번들의 인덱스(`cur`)를 동일 평가 질문 세트(12문항)로 비교합니다.
   * **첫 출시:** 최소 적중 수(`MIN_HIT_FIRST = 9`) 이상일 때만 합격
   * **이후 업데이트:** 기존 버전이 맞히던 문항을 새 버전이 틀리는 **품질 회귀(`flips`)가 0개**일 때만 승격 허용

### ⑤ 실패 전략의 3단 분리 (`d68_weekly_refresh.py`)
`default_args`에 `retries=2`, `retry_delay=3초`를 설정하되, 예외 종류에 따라 다르게 반응하도록 설계했습니다.

| 예외 유형 | 발생 상황 | Airflow 동작 | 설계 의도 |
| :--- | :--- | :--- | :--- |
| **일반 예외** (`DriveUnavailable`) | 공유 드라이브 일시 응답 없음 | **자동 재시도** (최대 2회) | 일시적 네트워크 장애 복구 기대 |
| **`AirflowFailException`** | `MissingSnapshot`, `BadConfig`, 통과율 미달, `flip` 발생 | 재시도 무시 후 **즉시 `failed`** | 다시 해도 실패할 오류는 시간 낭비 없이 차단 |
| **`AirflowSkipException`** | 백필 시 `gate`, 리허설 시 `promote`, 동일 번들 재출시 | 에러가 아닌 **정상 `skipped`** | 의도된 생략(운영 번들 보호) |

* **DAG Factory & `Param`:** 하나의 `make_refresh()` 함수로 운영용 `d68_weekly_refresh`(`is_paused_upon_creation=True`)와 UI 연습용 `d68_ops_rehearsal`(결과는 `ops_rehearsal/`에 분리, `promote`는 `skipped`)을 생성하며, `Param`과 `AIRFLOW__CORE__DAG_RUN_CONF_OVERRIDES_PARAMS="True"`를 통해 UI Trigger 시 설정값을 안전하게 오버라이드합니다.

---

## 6. 테스트 및 Airflow UI 실습·트러블슈팅 기록

### ① 사전 E2E 테스트 (`dag_obj.test()`)
* `run_one.py`로 프로세스를 격리해 `2026-09-28` 기준 데이터(`BASELINE`, `min_chars=10`)를 실행한 결과, 6개 태스크(`catalog -> prepare -> build_index -> evaluate -> gate -> promote`)가 모두 `success`로 완료되어 최초 운영 번들(`releases/current.json` → `r-20260928-<해시>`)이 정상 등록되었습니다.

### ② Airflow UI 시나리오 실습 (`d68_ops_rehearsal`, 입력 `snapshots/2026-11-09.json`)

| 단계 | 실행 조건 / 작업 | `run_id` | 태스크 상태 & 로그 근거 | 핵심 인사이트 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 잘못된 설정 Trigger** | `min_chars=150`으로 수동 실행 | `manual__2026-11-26T06:20:29.490432+00:00` | `catalog`~`evaluate`: `success`<br>`gate`: **`failed`** (`flip 2개 ['q11', 'q12']`)<br>`promote`: **`upstream_failed`** | 약 88자인 `출장여비규정.md`가 150자 미만으로 필터링되어 관련 질문(`q11`, `q12`)이 틀림. `gate`가 `AirflowFailException`으로 재시도 없이 즉시 차단해 운영 보호 |
| **2. `gate` 단독 Clear** | Upstream/Downstream 끄고 `gate` 1개만 Clear (`Affected Tasks: 1`) | `manual__2026-11-26T06:20:29.490432+00:00` | `catalog`~`evaluate`: 재실행 안 됨<br>`gate` (**Try 2**): 동일한 `flip 2개 ['q11', 'q12']`로 **`failed`** | **`Clear`는 상태만 초기화할 뿐 `Run Config(min_chars=150)`와 이미 생성된 상류 산출물을 바꾸지 못함** |
| **3. 원인 수정 후 새 Trigger** | `min_chars=10`으로 새 실행 생성 | `manual__2026-11-26T07:07:36.811066+00:00` | `catalog`~`gate`: **`success`**<br>`promote`: **`skipped`**<br>전체 DAG Run: **`success`** | 새 Run이 수정된 기준으로 처음부터 다시 계산해 `q11`, `q12`가 복구됨. `promote`는 리허설 규칙에 의해 의도적 건너뜀 처리되어 기존 운영 번들 유지 |

### ③ 상태 및 복구 방식 핵심 구분
* **`upstream_failed` vs `skipped`:** `upstream_failed`는 앞선 상류 태스크(`gate`)가 실패해 실행 기회조차 얻지 못하고 막힌 실패 연쇄 상태이고, `skipped`는 코드가 `AirflowSkipException`으로 의도적으로 건너뛴 정상 종료 상태입니다.
* **복구 원칙 (`RECOVERY`):** 같은 입력 스냅샷(`snapshots/YYYY-MM-DD.json`)과 원문(`blobs/<sha256>`)은 재사용하되, 정제 기준(`min_chars` 등)이 변경되면 파생 산출물(`prepare`의 `chunks.json` → `build_index`의 `index.json` → `evaluate` 보고서)을 새 실행으로 다시 계산해 품질 게이트를 검증합니다.

---

## 7. 실무 주의사항 & 안전한 종료 절차

### 실무 체크리스트
1. **Top-level 무거운 코드 금지:** DAG 파일 최상단에 DB 연결이나 외부 API 호출을 두면 DAG Processor의 주기적 파싱 때마다 실행되어 부하를 유발하므로 반드시 `@task` 함수 내부에 작성합니다.
2. **멱등성(Idempotency) 보장:** 같은 날짜를 여러 번 재실행해도 결과가 깨지지 않도록 스냅샷 고정, 버전별 디렉토리 분리, 원자적 쓰기를 활용합니다.
3. **재현 가능한 감사 추적(Audit Trail):** 다른 팀원이 동일한 판단을 재검증하려면 DAG 그래프 구조뿐만 아니라 `run_id`, `Run Config`, XCom 경로, 태스크 시도 횟수 및 로그, `current.json` 번들 정보를 함께 보관해야 합니다 (`d68_submission.json`).

### 안전한 환경 종료 순서
1. 노트북에서 최종 제출 셀을 실행해 `/workspace/work_d68/d68_submission.json` 생성 확인
2. 노트북에서 `stop_airflow()`를 실행해 컨테이너 내부 Airflow 프로세스 그룹 종료
3. 노트북 저장 후 로컬 PC에서 `work_d68/d68_submission.json` 및 `logs/` 확인
4. **호스트 PC 터미널**의 `d68_local` 폴더에서 아래 명령어로 Docker 컨테이너 종료:
   ```bash
   docker compose down
   ```

---

## 8. 참고 자료 & 다음 학습 목표

* [Airflow 3.3.2: Grid·Graph·Logs·XCom 화면](https://airflow.apache.org/docs/apache-airflow/3.3.2/ui.html)
* [Airflow 3.3.2 TaskFlow](https://airflow.apache.org/docs/apache-airflow/3.3.2/tutorial/taskflow.html)
* [Airflow 공식 Docker 이미지 확장](https://airflow.apache.org/docs/docker-stack/build.html)
* **다음 학습 과제:** 정기 스케줄 자동 실행 검증 · `airflow backfill` CLI 운영 · `Asset` 이벤트 기반 다중 DAG 트리거(`write_release_note` 연동)
