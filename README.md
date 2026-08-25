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
