# 🚀 나만의 AIFFEL 학습 저장소
구글 코랩과 VS Code를 연동하여 딥러닝을 공부하는 공간입니다.

### 📅 학습 일지
2026 08 19
개발 환경 세팅 (코랩 + VS Code + 깃허브 연동 완료)

2026 08 25
#### 머신러닝의 알고리즘 종류는 크게 3가지
  지도학습 (Supervised Learning)
  비지도학습 (Unsupervised Learning)
  강화학습 (Reinforcement Learning)

강화학습은 앞에서 언급한 지도학습, 비지도학습과는 다른 종류의 알고리즘
에이전트(Agent): 학습 주체 (혹은 actor, controller)
환경(Environment): 에이전트에게 주어진 환경, 상황, 조건
행동(Action): 환경으로부터 주어진 정보를 바탕으로 에이전트가 판단한 행동
보상(Reward): 행동에 대한 보상을 머신러닝 엔지니어가 설계

#### 강화학습 알고리즘의 대표적인 종류
  Monte Carlo methods
  Q-Learning
  Policy Gradient methods



# 📌 Machine Learning 기초 학습 정리

## 1. 사이킷런(`sklearn.datasets`) 데이터셋 구조
- **Toy Dataset (Loaders)**: 용량이 작고 내장되어 있어 `load_데이터이름()`으로 즉시 불러옴 (예: `load_iris()`, `load_wine()`)
- **Real World Dataset (Fetchers)**: 용량이 크며 `fetch_데이터이름()`을 통해 다운로드 후 불러옴

---

## 2. 데이터 구조와 차원 (Shape) 규칙
- **특성 행렬 ($X$, Input)**: 반드시 **2차원 행렬** 형태 `(데이터 개수, 특성 개수)`
- **타겟 벡터 ($y$, Target)**: **1차원 벡터** 형태 `(데이터 개수,)`
- **핵심 규칙**:
  - 행(Row) = 데이터/샘플의 개수
  - 열(Column) = 특성(Feature)의 개수
  - 특성이 $N$개인 데이터를 처리하는 모델의 입구 차원은 `(데이터 개수, N)`으로 고정됨.

---

## 3. `reshape()` 함수와 `-1` 기법
- `reshape()`에서 **`-1`은 한 번에 딱 하나만 사용 가능** (컴퓨터가 나머지 차원을 자동 계산).
- **`reshape(-1, 1)`**: 열을 1개로 고정하고 행 개수를 자동 맞춰줌 (1차원 벡터 $\rightarrow$ 2차원 특성 행렬 변환 시 활용).
- **`reshape(-1, N)`**: 특성이 $N$개일 때 행의 개수 변화에 유연하게 대응 가능.
- **주의**: `reshape(-1, -1)`은 기준점이 없어 계산이 불가능하므로 **에러 발생**.

---

## 4. 데이터 분할과 평가 (치팅 / 과적합 문제)
- **학습 데이터 = 평가 데이터 일치 시 문제점**:
  - 모델이 학습용 문제와 정답을 통째로 암기하여 **정확도 100%(1.0)가 나오는 착시 현상** 발생.
- **해결책**:
  - 반드시 데이터를 **공부용(Train Data)**과 **시험용(Test Data)**으로 쪼개서(`train_test_split`) 한 번도 보지 못한 새로운 데이터로 모델의 진짜 실력을 평가해야 함.

---

## 5. 머신러닝 $\rightarrow$ 딥러닝 $\rightarrow$ LLM의 공통 철칙
- **입력 차원의 일치 (Dimensionality Match)**:
  - 모델 학습 시 사용한 **특성 개수(열 개수)**와 예측 시 입력하는 **데이터의 특성 개수**는 $1:1$로 완전히 일치해야 함.
  - 특성 개수가 다르면 머신러닝, 딥러닝, LLM 모두 수학적 연산 불가능으로 인해 **`ValueError`** 발생.

 ## 📌 Scikit-Learn 주요 모듈 정리

| 구분 | 모듈명 | 설명 |
| :--- | :--- | :--- |
| **데이터셋** | `sklearn.datasets` | 사이킷런에서 기본 제공하는 예제 데이터셋 |
| **데이터타입** | `sklearn.utils.Bunch` | 사이킷런 데이터셋의 기본 데이터 타입 (딕셔너리와 유사한 구조) |
| **데이터 전처리** | `sklearn.preprocessing` | 데이터 전처리 기능 제공 (정규화, 인코딩, 스케일링 등) |
| **데이터 분리** | `sklearn.model_selection.train_test_split` | 학습용(Train) 및 테스트용(Test) 데이터셋 분리 |
| **평가** | `sklearn.metrics` | 분류, 회귀, 클러스터링 알고리즘의 성능 평가 지표 제공 |
| **머신러닝 알고리즘** | `sklearn.ensemble` | 앙상블 알고리즘 (Random Forest, AdaBoost, Gradient Boosting 등) |
| | `sklearn.linear_model` | 선형 모델 알고리즘 (Linear Regression, Ridge, Lasso, SGD 등) |
| | `sklearn.naive_bayes` | 나이브 베이즈(Naive Bayes) 분류 알고리즘 |
| | `sklearn.neighbors` | 최근접 이웃 알고리즘 (KNN 등) |
| | `sklearn.svm` | 서포트 벡터 머신(Support Vector Machine) 알고리즘 |
| | `sklearn.tree` | 의사결정 트리(Decision Tree) 기반 알고리즘 |
| | `sklearn.cluster` | 군집화(Clustering) 알고리즘 (K-Means 등) |
