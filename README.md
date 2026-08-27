# 🚀 나만의 AIFFEL 학습 저장소
구글 코랩과 VS Code를 연동하여 딥러닝을 공부하는 공간입니다.

### 📅 학습 일지
2026 08 19
개발 환경 세팅 (코랩 + VS Code + 깃허브 연동 완료)
---
2026 08 25 [머신러닝,결측치,이상치,정규화,스케일링](https://github.com/orthia194/aiffel-study/blob/main/2026/08/25/README.md)
---
2026 08 26 [머신러닝 평가 지표 핵심 정리](https://github.com/orthia194/aiffel-study/blob/main/2026/08/26/README.md)
---
2026 08 27 [Regularization](https://github.com/orthia194/aiffel-study/blob/main/2026/08/27/README.md)
---

---
### 📖 머신러닝 / 딥러닝 핵심 용어 사전

#### 1. 데이터 전처리 (Data Preprocessing)
* **결측치 (Missing Value):** 데이터셋에서 누락되거나 비어있는 값 (예: NaN, Null).
* **이상치 (Outlier):** 일반적인 데이터 범주에서 크게 벗어난 비정상적인 값.
* **원-핫 인코딩 (One-Hot Encoding):** 범주형 문자 데이터를 [1, 0, 0]과 같이 0과 1의 이진 벡터로 변환하는 작업.
* **정규화 (Normalization):** 서로 다른 데이터의 스케일(범위)을 $0 \sim 1$ 등의 동일한 구간으로 맞추는 과정.
* **특징 / 피처 (Feature):** 모델 예측의 입력으로 사용되는 데이터의 속성이나 변수 (예: 집값 예측의 '평수', '방 개수').
* **라벨 / 타겟 (Label / Target):** 모델이 예측해야 하는 정답 데이터.

#### 2. 모델 학습 및 단위 (Training & Units)
* **가중치 (Weight, $W$):** 입력 데이터가 결과에 미치는 영향력을 나타내는 학습 파라미터.
* **에포크 (Epoch):** 전체 학습 데이터를 처음부터 끝까지 1회 완독하는 학습 단위.
* **배치 사이즈 (Batch Size):** 모델이 한 번에 묶어서 학습하는 데이터의 개수.
* **스텝 / 이터레이션 (Step / Iteration):** 1 에포크를 완주하기 위해 배치를 제출 및 학습한 총 횟수.
* **학습률 (Learning Rate):** 가중치를 한 번 수정할 때 이동하는 변화의 크기(발걸음 폭).

#### 3. 성능 평가 및 오차 (Evaluation & Loss)
* **손실 함수 (Loss Function):** 모델의 예측이 틀린 정도를 계산하여 학습의 방향을 잡아주는 수치.
* **평가지표 (Metric):** 사람(개발자)이 모델의 최종 성적을 한눈에 판단하기 위해 사용하는 지표 (예: Accuracy, RMSE).
* **MSE (Mean Squared Error):** 예측값과 실제값 차이의 제곱 평균.
* **RMSE (Root Mean Squared Error):** MSE에 루트를 씌워 실제 데이터와 동일한 단위로 맞춘 평가지표.
* **MAE (Mean Absolute Error):** 오차의 절대값을 평균 낸 지표로, 이상치 영향에 강함.
* **정확도 (Accuracy):** 전체 데이터 중 모델이 정답을 맞춘 비율.

#### 4. 학습 상태 및 최적화 (Optimization)
* **과적합 (Overfitting):** 모델이 훈련 데이터의 잡음까지 외워버려, 새로운 테스트 데이터에서 성적이 떨어지는 현상.
* **과소적합 (Underfitting):** 모델이 데이터의 패턴을 충분히 학습하지 못해 훈련 데이터에서도 성적이 낮게 나오는 현상.
* **규제 (Regularization):** 과적합을 막기 위해 가중치 크기에 제약을 가하거나 노드를 지우는 기법 (예: L1, L2, Dropout).
* **배치 정규화 (Batch Normalization):** 딥러닝 레이어 사이에서 배치 단위 데이터 분포를 정돈하여 학습을 안정화하는 기술.
* **조기 종료 (Early Stopping):** 검증 오차(Validation Loss)가 올라가는 지점(과적합 시작점)에서 학습을 자동으로 중단하는 기법.
