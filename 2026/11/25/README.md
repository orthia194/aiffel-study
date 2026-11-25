# 🐳☸️ Docker & Kubernetes 기반 MLOps 모델 서빙 정리

AI 모델(QA API 및 LLM)을 컨테이너화하고 Kubernetes(`kind`) 위에서 **안정적으로 배포·확장·복구·보안 검증**하는 과정을 개념부터 실전 트러블슈팅까지 정리한 학습 기록입니다.

---

## 1. 핵심 개념 정리 (Docker · Kubernetes · MLOps)

### 1.1 Docker와 Kubernetes의 관계, 그리고 MLOps에 필수인 이유
* **Docker (표준 컨테이너 포장 기술):** 애플리케이션 코드와 실행 환경(Python, PyTorch, CUDA 버전 등)을 하나의 표준 이미지로 패키징하여 *"내 컴퓨터에선 됐는데 서버에서 안 돼요"* 문제를 제거합니다.
  *(참고: 정확히는 Docker가 만든 OCI 표준 이미지를 Kubernetes 내부 런타임인 `containerd` 등이 실행·관리합니다.)*
* **Kubernetes (컨테이너 오케스트레이션):** 수십~수백 개의 컨테이너를 여러 서버에 걸쳐 **자동 배치(스케줄링), 장애 시 자동 복구(Self-healing), 트래픽 분산(Load Balancing), 무중단 버전 교체 및 롤백**을 수행하는 총괄 관리자입니다.
* **왜 MLOps에서 필수일까?**
  1. **복잡한 의존성 & 재현성 해결:** 학습 환경과 서빙 환경의 Python/CUDA/라이브러리 버전을 이미지로 동일하게 고정.
  2. **단계별 파이프라인 분리:** 데이터 전처리(CPU) ➔ 모델 학습(GPU) ➔ 서빙 등 각 단계를 독립 컨테이너(`Pod`)로 실행.
  3. **비싼 GPU 자원의 효율적 풀링(Pooling):** 여러 GPU 노드를 하나의 풀로 묶어, 작업이 들어올 때만 GPU를 할당하고 완료 시 즉시 반납.
  4. **탄력적 모델 서빙 & 무중단 배포:** 트래픽 급증 시 자동 복제(Auto-scaling), 모델 성능 저하 시 이전 버전으로 즉시 롤백(`rollout undo`).

---

### 1.2 Kubernetes 클러스터 구조와 핵심 객체
Kubernetes는 **사령부(Control Plane)**와 **일꾼 서버(Worker Node)**를 네트워크로 묶어 하나의 거대한 컴퓨터처럼 제어합니다.

* **Control Plane (사령부):** 스케줄링·상태 감시·배포 판단을 내리는 두뇌 (`kubectl` 리모컨으로 명령 전달).
* **Worker Node & `kubelet` (일꾼 서버 & 현장 반장):** 각 서버에 설치된 에이전트(`kubelet`)가 사령부의 지시를 받아 해당 서버의 컨테이너 엔진으로 컨테이너를 직접 실행·감시.
* **핵심 3대 객체의 연결 구조:**
  * **`Pod` (완두콩 껍질):** Kubernetes가 IP와 볼륨을 부여하는 최소 관리 단위. 다양한 컨테이너 엔진을 추상화하고, 메인 컨테이너 옆에 로그/모니터링 보조 컨테이너(Sidecar)를 한 몸처럼 묶을 수 있음.
  * **`Deployment` (Pod 관리자):** `Pod`의 개수 유지(`replicas`), 자동 복구, 롤링 업데이트 및 롤백을 책임짐.
  * **`Service` (변하지 않는 고정 창구):** `Pod`는 죽고 새로 뜰 때마다 IP가 바뀌므로, **`Label`/`Selector`(이름표)**를 보고 건강한 `Pod`들에게만 트래픽을 분산해 주는 고정 주소/로드밸런서.

```text
[사용자 / Gateway] ──▶ [ Service (고정 이름·IP, 예: http://qa:8000) ]
                           │  (selector: app=qa & Ready 상태인 Pod에만 분산)
                           ▼
                  ┌────────┴────────┐
                  ▼                 ▼
              [ Pod 1 ]         [ Pod 2 ]     ◀── [ Deployment ] (개수 유지 & 롤링·롤백)
             (app: qa)         (app: qa)
```

---

### 1.3 도커 이미지(`COPY`) vs 볼륨 마운트(`Mount`) 분리 원칙
모든 파일을 볼륨 마운트로 연결하면 **로컬 개발 시엔 편하지만(Hot-reload)**, 운영/Kubernetes 환경에서는 **① 이미지 이식성 상실 ② 다중 서버 스케일링 불가 ③ 롤링 업데이트 및 롤백 시 구버전/신버전 코드 충돌**이 발생합니다.

> **💡 볼륨 마운트의 오해 바로잡기:** 일반 바인드 마운트(`-v`)는 파일을 양쪽에 복제하는 것이 아니라 **호스트 폴더를 유리창 너머로 직접 바라보는 포인터**이며, 이미지 안에 있던 기존 폴더 내용을 가려버립니다. (개발 중 이미지 구조를 유지하며 변경분만 실시간 복제하려면 `docker compose watch`의 `sync` 기능을 사용합니다.)

| 구분 | 포함할 대상 | 이유 및 실전 패턴 |
| :--- | :--- | :--- |
| **이미지 내부 (`COPY`)**<br>*(불변 완제품)* | • 앱 소스 코드 (`app.py`, `serve.py`)<br>• 패키지 (`requirements.lock`)<br>• 수백 MB 이하의 경량 ML 모델 | 버전(`v1`, `v2`)별로 코드가 박제되어 어느 노드에서든 동일하게 실행되고 깔끔한 **롤백**이 가능 |
| **외부 볼륨 / K8s 리소스**<br>*(가변·대용량·비밀)* | • **초대형 LLM 가중치 (7B≈14GB, 32B≈64GB):** `PVC(ReadOnlyMany)` 공유 볼륨 마운트, 또는 `initContainer`로 S3/GCS에서 노드 로컬 NVMe SSD(`emptyDir`)로 고속 다운로드<br>• **영속 데이터/로그:** `PVC` 또는 `emptyDir`<br>• **설정/비밀:** `ConfigMap`, `Secret` | 이미지 비대화(수십 GB)로 인한 오토스케일링 지연 방지, 컨테이너 재시작 시 데이터 유실 방지, 보안 키 유출 차단 |

---

## 2. 실무 핵심 노하우 & 트러블슈팅 가이드

### 2.1 GPU 롤링 업데이트 데드락(Deadlock) 원인과 해결법

#### 💥 왜 GPU 환경에서만 데드락이 잘 걸릴까?
* Kubernetes의 `nvidia.com/gpu: "1"`은 GPU 번호(`0번`, `1번`)가 아니라 **독점 할당할 GPU 수량(`1장`)**을 의미합니다. (기본 규칙상 소수점 `0.5` 할당 불가)
* `Deployment`의 기본 롤링 전략은 `maxSurge: 25%`(올림 `ceil`), `maxUnavailable: 25%`(내림 `floor`)입니다.
* **`replicas`가 1~3대일 때:** `floor(replicas × 0.25) = 0`이 되어 **`maxSurge=1, maxUnavailable=0`**(*"기존 파드를 단 1개도 먼저 끄지 말고 새 파드부터 띄워라"*)으로 동작합니다.
* **교착 상태(Deadlock) 발생:** 여유 GPU가 `0장`일 때 새 파드는 GPU를 못 받아 **`Pending`**에 갇히고, 기존 파드는 새 파드가 `Ready`가 될 때까지 GPU를 쥔 채 안 꺼지므로 **영원히 멈췄다가 `ProgressDeadlineExceeded`로 실패**합니다.

#### ✅ 실무 상황별 해결 전략 3가지
1. **여유 GPU가 없고 잠깐 중단이 허용되는 환경 / 1-GPU 가정·개발 PC:**
   * **`maxSurge: 0, maxUnavailable: 1`**로 설정합니다. (기존 파드를 먼저 1개 내려 GPU를 반납한 뒤 새 파드를 기동 ➔ `replicas=1`이면 로딩 시간 동안 일시 중단되지만, `replicas=3` 이상이면 2대가 계속 요청을 받으며 무중단 교체 가능!)
2. **1초도 끊기면 안 되는 상용 AI 서비스 (무중단 필수):**
   * 클러스터에 **최소 1장 이상의 예비 GPU**를 두거나, 클라우드 **Cluster Autoscaler**와 연동해 업데이트 동안만 임시 GPU 노드를 1대 증설한 뒤 교체 완료 후 반납합니다.
3. **경량 모델 여러 개를 GPU 1장에 띄워야 할 때:**
   * 일반 Docker(`--gpus all`)는 VRAM 한도 내에서 동시 실행이 가능하며, Kubernetes에서는 NVIDIA GPU Operator의 **Time-Slicing(시분할)** 또는 **MIG(데이터센터 GPU 하드웨어 분할)**를 사용합니다. *(단, 합산 VRAM 초과 시 `OOM`으로 동시 사망하므로 대형 모델 교체 시엔 1번 전략이 정석)*

---

### 2.2 AI 모델 서빙의 3종 Probe & 기동·리소스 설계

* **`httpGet` 프로브의 함정:** Kubernetes 프로브는 JSON 본문(`{"status": "loading"}`)을 읽지 않고 **오직 HTTP 상태 코드(`200~399`)만 판별**합니다.
  * `startupProbe` (`/health`): 무거운 모델이 GPU에 올라갈 때까지(`D67_LOAD_FIRST=1`처럼 포트 오픈이 늦는 `vLLM` 등 포함) 충분한 유예 시간(`periodSeconds × failureThreshold > 모델 로딩 시간`)을 주어 **`CrashLoopBackOff` 무한 재시작 방지**.
  * `readinessProbe` (`/ready`): 로딩 중에는 반드시 `503`을 반환하게 하여, `Service`가 준비 안 된 파드로 사용자 트래픽을 보내지 않도록 차단.
  * `livenessProbe` (`/health`): 기동 완료 후 프로세스 먹통 여부 감시(먹통 시 자동 재시작).
* **vLLM / PyTorch 필수 볼륨 (`/dev/shm`):** 컨테이너 기본 공유 메모리(`/dev/shm`)는 **64MB**로 제한되어 있어 멀티 프로세스/텐서 병렬 처리 시 `Bus error`로 사망합니다. `emptyDir (medium: Memory, sizeLimit: 2Gi)`를 `/dev/shm`에 마운트해 해결합니다.
* **`resources` 설계 (`requests` vs `limits`):**
  * `requests` (`cpu`, `memory`): 스케줄러가 파드를 배치할 빈 노드 공간을 계산하는 기준.
  * `limits.memory`: 메모리 누수·대용량 입력 폭주 시 해당 파드만 **`OOMKilled`**로 정리해 동일 노드 내 다른 AI 서비스를 보호. *(CPU `limits`는 스로틀링으로 지연 시간이 튈 수 있어 의도적으로 비우기도 함)*

---

### 2.3 프로덕션 컨테이너 & 보안·공급망 체크리스트

1. **재현성 & 경량화:** 베이스 이미지 다이제스트(`@sha256:...`) 고정, Multi-stage Build + `python -m venv --without-pip`로 런타임에서 `pip`까지 제거.
2. **레이어 캐싱 순서:** 무거운 패키지 설치(`pip install`, 예: `vLLM`의 `5.11 GB` 레이어)를 위에 두고, 자주 바뀌는 소스 코드(`COPY qa/`)는 맨 아래에 배치.
3. **Non-root & 숫자 UID:** `kubelet`은 `runAsNonRoot: true` 검사 시 `/etc/passwd`의 문자 이름을 읽지 못하므로 반드시 **`USER 10001:10001`**처럼 숫자 지정. 코드 폴더는 `root` 소유 + `readOnlyRootFilesystem: true`로 잠그고, 쓰기 필요한 경로만 `emptyDir` 마운트 (`PermissionError` 방지).
4. **종료 신호 (`SIGTERM`):** `CMD ["python", "-m", "qa.serve"]` 같은 Exec 형식을 써야 `/bin/sh`가 `SIGTERM`을 가로채지 않고 서버에 즉시 전달해 30초 대기 없이 우아한 종료(`Graceful Shutdown`) 가능.
5. **시크릿 유출 & 화이트아웃(`.wh.`) 함정:** 이미지 레이어는 누적되므로 `COPY . .` 뒤 `RUN rm -f .env`를 해도 **하위 레이어에 키가 그대로 남고 `.wh..env` 가림막만 생김**. 반드시 `.dockerignore`로 `**/.env`, `keys/`를 원천 차단하고, `ARG`/`ENV`에도 키를 넣지 않으며, 사내망 반입 전 전 레이어 스캔 수행.
6. **의존성 고정 & SBOM & 서명 키 교체:** `pip --dry-run --report`로 간접 의존성까지 `requirements.lock`에 잠그고 `CycloneDX` SBOM으로 취약점 추적 기반 확보. 폐쇄망 반입 시 `Ed25519` 개인키 서명 + 공개키 검증을 수행하며, 키 교체 시 `{"2026-old": ..., "2026-new": ...}` ➔ `{"2026-new": ...}` 2단계로 무중단 교체(Key Rotation).

---

## 3. 오늘의 실습 코드 검증 및 실행 결과

### 3.1 오프라인 검증 & 시뮬레이션 결과 (Google Colab)

| 실습 항목 | 검증 내용 | 실제 결과 |
| :--- | :--- | :--- |
| **이미지 체급 & 레이어 분석** | Docker Hub API 조회 (`vllm/vllm-openai:v0.30.0` vs `slim`) | `python:3.12-slim` **0.04 GB** vs `cuda-runtime` **1.65 GB** vs `cuda-devel` **3.97 GB** vs `vllm` **8.73 GB** (`pip` 패키지 75.1% · 단일 최대 레이어 **5.11 GB**) |
| **Dockerfile & K8s YAML 정적 검사** | `hadolint`, `check_stages`, `kubernetes-validate (v1.36.0)`, `check_manifests` | `qa.yaml`·`gateway.yaml` 규격 오류 **0건** 및 보안/리소스/Probe 규칙 모두 통과 (`Dockerfile.d50`은 다이제스트·root 미설정 지적) |
| **Readiness Probe 경로 실험** | `D67_WARMUP=4` 새 파드 투입 중 `0.2초` 간격 트래픽 전송 | `/health` 잘못 지정 시 1.3초 만에 조기 투입되어 **503 에러 9건** (`{200: 19, 503: 9}`) ➔ `/ready` 지정 시 5.7초 후 투입되어 **에러 0건** (`{200: 34}`) |
| **Startup Probe 실험** | 포트 오픈 지연(`D67_LOAD_FIRST=1`, 로딩 6초) 환경에서 12초 관찰 | `startup` 없음 **재시작 5번** / 5초 설정 **재시작 2번** (로딩 실패) ➔ **15초 설정 재시작 0번, 7.3초 만에 기동 성공** |
| **Non-root 쓰기 권한 실험** | `root` 소유 `/app`에서 `10001` 유저로 `logs/` 생성 시도 | 미마운트 시 **`PermissionError` 발생** ➔ `10001` 소유 볼륨 마운트 시 정상 기동 |
| **GPU 롤링 수학 (`rollout_plan`)** | 빈 GPU 0장, 기본 `25%` 전략으로 `replicas`별 롤링 시뮬레이션 | **2대·3대: `멈춤`(데드락)** / **4대·8대: `끝남`** (`floor` 값이 4대부터 1이 됨). 3대도 `maxSurge=0, maxUnavailable=1` 적용 시 매 스텝 2대 유지하며 **4스텝 만에 `끝남`** |
| **Compose 헬스체크 튜닝** | 8초 예열 서버에 기본값(`interval 30s`) vs 조정값(`start_period 60s, start_interval 1s`) | 기본값 **`30.1초`** ➔ 조정값 **`10.1초`** (약 20초 단축) |
| **이미지 시크릿 스캐너** | `.env` 복사 후 `rm -f .env` + `ARG`/`ENV` 유출 모의 이미지(`qa-leaky-practice.tar`) 스캔 | 레이어 1 원본 + `.wh..env` 화이트아웃 + `Env`/`history` 포함 **총 5건 탐지 (종료 코드 1)** |
| **의존성 Lock & SBOM & 서명** | `requirements.txt` ➔ `requirements.lock` ➔ `CycloneDX 1.6` ➔ `Ed25519` 서명·변조 테스트 | 직접 명시 **4줄 ➔ 실제 패키지 19개** 고정, SBOM 19개 생성, 파일/명세서 변조 시 반입 거절 및 2단계 키 교체(`during` ➔ `after`) 검증 |

---

### 3.2 실제 Kubernetes 클러스터 실행 결과 (Google Cloud Shell `kind-d67`)

> 실행 환경: `Docker 29.9.3` / `kind 0.33.0` / `kubectl v1.35.9`

| 실전 테스트 | 실행 내용 | 관찰된 실제 결과 |
| :--- | :--- | :--- |
| **최초 배포 & 예열** | `qa-server:1.5.0` 빌드 및 `kubectl apply -f k8s/qa.yaml` | 초기 `0/1 Running` 상태에서 8초 워밍업 후 **`10초` 만에 `1/1 Running` 전환**, `port-forward` ➔ FastAPI `/docs` `/ask` 정상 응답 |
| **자가 치유 & 스케일링** | 실행 중 파드(`k2z94`) 강제 삭제 및 `replicas=4` 확장 | 삭제 파드는 `SIGTERM` 수신 후 **1초 내 `Completed` 종료**, 새 파드(`hbclw`) 즉시 생성 후 **`10초` 만에 `1/1`** 복구. 스케일링 시 4개 파드 즉시 생성 |
| **롤링 업데이트 & 롤백** | `--build-arg VERSION=2.0.0` 빌드 ➔ `set image` ➔ `rollout undo` | 레이어 캐싱으로 **`0.5s` 만에 빌드**, 새 파드(`11~12s`에 `1/1`) 확인 후 옛 파드 종료(무중단), `/ready` `"판":"2.0.0"` 확인 후 `undo`로 `"판":"1.5.0"` 즉시 복구 |
| **메모리 상한 테스트** | `limits.memory: 128Mi` 파드(`oom-test`)에서 `300Mi` 할당 | 생성 **`9s` 만에 `OOMKilled`** 종료 확인 (다른 정상 파드 영향 없음) |
| **GPU 데드락 재현 & 해소** | 노드에 가짜 GPU `example.com/gpu: 2` 패치 ➔ 2파드 점유 후 `1.6.0` 업데이트 ➔ `2.0.0` + `maxSurge=0, maxUnavailable=1` 적용 | `1.6.0` 새 파드가 **`Pending`** (`FailedScheduling: 1 Insufficient example.com/gpu`) 상태로 멈추고 60초 후 **`ProgressDeadlineExceeded`** 기록 ➔ 전략 수정 즉시 기존 파드 1개씩 `Completed` 후 **`13s`·`11s` 만에 `1/1` 교체 완료** |

---

### 3.3 주요 실행 명령어 레퍼런스 (`Cloud Shell / kind`)

```bash
# 1. 이미지 빌드 · 클러스터 생성 · 이미지 적재
docker build -t qa-server:1.5.0 .
kind create cluster --name d67
kind load docker-image qa-server:1.5.0 --name d67

# 2. 배포 및 8초 예열 관찰 (0/1 -> 1/1)
kubectl apply -f k8s/qa.yaml
kubectl get pods -w

# 3. 자가 치유 & 스케일링 & 서비스 접속 확인
kubectl delete pod <파드_이름> --wait=false
kubectl scale deployment qa --replicas=4
kubectl port-forward svc/qa 8000:8000

# 4. 레이어 캐시 활용 버전 업(2.0.0) & 롤백(1.5.0)
kubectl scale deployment qa --replicas=2
docker build --build-arg VERSION=2.0.0 -t qa-server:2.0.0 .
kind load docker-image qa-server:2.0.0 --name d67
kubectl set image deployment/qa qa=qa-server:2.0.0
kubectl exec deploy/qa -- python -B -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/ready').read().decode())"
kubectl rollout undo deployment/qa
kubectl rollout status deployment/qa

# 5. 가짜 GPU 데드락 재현 및 해소
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl patch node "$NODE" --subresource=status --type=json -p='[{"op":"add","path":"/status/capacity/example.com~1gpu","value":"2"}]'
# ... (deployment에 example.com/gpu: 1, strategy 25%, progressDeadlineSeconds: 60 패치 후 set image로 데드락 확인)
kubectl patch deployment qa --type=strategic -p '{"spec":{"strategy":{"rollingUpdate":{"maxSurge":0,"maxUnavailable":1}}}}'

# 6. 실습 종료 후 클러스터 정리
kind delete cluster --name d67
```
