# 🚀 나만의 AIFFEL 학습 저장소
구글 코랩과 VS Code를 연동하여 딥러닝을 공부하는 공간입니다.

### 📅 학습 일지
2026 08 19
개발 환경 세팅 (코랩 + VS Code + 깃허브 연동 완료)  
2026 08 25 [머신러닝,결측치,이상치,정규화,스케일링](https://github.com/orthia194/aiffel-study/blob/main/2026/08/25/README.md)  
2026 08 26 [머신러닝 평가 지표 핵심 정리](https://github.com/orthia194/aiffel-study/blob/main/2026/08/26/README.md)  
2026 08 27 [Regularization](https://github.com/orthia194/aiffel-study/blob/main/2026/08/27/README.md)  
2026 09 01 [Deep Learning & Computer Vision Core Concepts](https://github.com/orthia194/aiffel-study/blob/main/2026/09/01/README.md)
2026 09 02 [자연어](https://github.com/orthia194/aiffel-study/blob/main/2026/09/02/README.md)

---

# 🚀 PyTorch Tensor Cheatsheet

파이토치(PyTorch) 기본 텐서 연산 및 핵심 개념을 정리한 요약집입니다.

---

## 📌 필수 개념 요약

* **Shape 읽는 법:** `(Batch, Channels, Height, Width)`
  * `(5,)` → **1차원** (원소 5개짜리)
  * `(4, 5)` → **2차원** (4행 5열)
  * `(32, 3, 224, 224)` → **4차원** (32장 배치, RGB 3채널, 224x224)
* **`dim` 차원 축소 연산:** `t.sum(dim=0)` 지정 시 **해당 차원(0번)이 접혀서 사라짐**
* **NumPy 연동 (`from_numpy`):** 메모리를 새로 복사하지 않고 **공유(Zero-copy)**하므로, 한쪽을 수정하면 다른 한쪽도 즉시 반영됨

---

## 🛠️ 주요 명령어 모음

### 1. 텐서 생성 (Creation)
| 명령어 | 설명 |
|---|---|
| `torch.tensor(data)` | 리스트나 배열을 텐서로 변환 |
| `torch.zeros(shape)` | 모든 원소가 `0`인 텐서 생성 |
| `torch.ones(shape)` | 모든 원소가 `1`인 텐서 생성 |
| `torch.rand(shape)` | 0~1 사이의 균등분포 난수 생성 |
| `torch.randn(shape)` | 평균 0, 표준편차 1인 정규분포 난수 생성 |
| `torch.from_numpy(array)` | NumPy 배열을 메모리 공유 텐서로 변환 |

### 2. 모양 및 차원 변경 (Reshaping)
| 명령어 | 설명 |
|---|---|
| `t.shape` / `t.size()` | 텐서의 모양(크기) 확인 |
| `t.view(...)` / `t.reshape(...)` | 텐서의 모양 변환 (`-1` 사용 시 자동 계산) |
| `t.squeeze()` | 크기가 1인 차원 제거 (예: `(1, 3, 224, 224)` → `(3, 224, 224)`) |
| `t.unsqueeze(dim)` | 지정한 위치에 크기가 1인 차원 추가 |

### 3. 연산 및 통계 (Math & Reduction)
| 명령어 | 설명 |
|---|---|
| `A * B` | 원소별 곱 (Element-wise Product) |
| `A @ B` | 행렬 곱 (Matrix Multiplication) |
| `t.sum(dim)` / `t.mean(dim)` | 지정한 차원 기준으로 합/평균 계산 |
| `t.max(dim)` / `t.argmax(dim)` | 최댓값 및 최댓값의 인덱스 위치 반환 |

### 4. 텐서 결합 (Combine)
| 명령어 | 설명 |
|---|---|
| `torch.cat([t1, t2], dim)` | 기존 차원을 따라 텐서 이어붙이기 |
| `torch.stack([t1, t2], dim)` | 새로운 차원을 만들어 텐서 쌓기 |

### 5. 장치 이동 (Device Control)
| 명령어 | 설명 |
|---|---|
| `t.to('cuda')` / `t.cuda()` | 텐서를 GPU(CUDA) 메모리로 이동 |
| `t.to('cpu')` / `t.numpy()` | CPU로 이동 후 NumPy 배열로 변환 |
---
# 📖 머신러닝 & 딥러닝 핵심 용어 통합 사전

---

## 1. 데이터 전처리 (Data Preprocessing)

* **특징 / 피처 (Feature, $X$):** 모델 예측의 입력으로 사용되는 데이터의 속성이나 변수 (예: 집값 예측의 '평수', '방 개수').
* **라벨 / 타겟 (Label / Target, $Y$):** 모델이 예측해서 맞춰야 하는 실제 정답 데이터.
* **결측치 (Missing Value):** 데이터셋에서 누락되거나 비어있는 값 (예: `NaN`, `Null`).
* **이상치 (Outlier):** 일반적인 데이터 범주에서 크게 벗어난 비정상적인 수치.
* **원-핫 인코딩 (One-Hot Encoding):** 범주형 문자 데이터를 `[1, 0, 0]`과 같이 0과 1의 이진 벡터 형태(범주형 데이터)로 변환하는 작업.
* **정규화 (Normalization):** 서로 다른 피처의 스케일(범위)을 $0 \sim 1$ 등의 동일한 구간으로 맞추어 학습 효율을 높이는 과정.

---

## 2. 학습 데이터 단위 및 횟수 (Data & Epoch)

* **Train Data Set (학습 데이터셋):** 모델을 학습시키기 위해 사용하는 전체 데이터의 집합.
* **배치 사이즈 / 미니 배치 (Batch Size / Mini Batch):** 모델이 한 번의 학습에서 묶어서 가져오는 데이터의 개수/규모.
* **스텝 / 이터레이션 (Step / Iteration):** 배치(Batch) 하나를 학습하고 가중치를 1회 업데이트하는 단위 횟수.
* **에포크 (Epoch):** 전체 학습 데이터셋을 처음부터 끝까지 한 번 모두 학습시킨 상태 (1 Epoch = 전체 데이터 완독).
  * *예시:* 전체 데이터가 100개이고 Batch Size가 10이면 $\rightarrow$ 10 Iteration = 1 Epoch.

---

## 3. 신경망 구조 및 연산 (Neural Network Architecture)

* **노드 / 뉴런 (Node / Neuron):** 인공신경망을 구성하는 가장 기본적인 계산 단위.
* **입력층 (Input Layer):** 모델로 입력되는 데이터($x_1, x_2$ 등)를 받아들이는 첫 번째 레이어.
* **은닉층 (Hidden Layer):** 입력층과 출력층 사이에 위치하며, 데이터의 복잡한 특징과 패턴을 추출하는 레이어.
* **출력층 (Output Layer):** 최종 예측값($\hat{y}$)을 계산하여 출력하는 마지막 레이어.
* **완전 연결 층 (Dense Layer / Fully Connected Layer):** 이전 레이어의 모든 노드가 다음 레이어의 모든 노드와 1:1로 전부 연결된 레이어 구조.
* **다층 퍼셉트론 (Multi-Layer Perceptron, MLP):** 하나 이상의 은닉층을 쌓아 만든 가장 대표적인 기본 딥러닝 인공신경망 구조.
* **가중치 (Weight, $w$):** 입력 값이 결과에 미치는 중요도나 영향력을 조절하는 학습 파라미터 (기울기).
* **편향 (Bias, $b$):** 노드가 얼마나 쉽게 활성화(발화)될지를 결정하여 그래프 전체의 위치를 이동시키는 상숫값 (절편, $y = wx + b$).
* **활성화 함수 (Activation Function):** 입력 값에 비선형성(Non-linearity)을 부여하여 신경망이 복잡한 패턴을 학습할 수 있게 만들어주는 함수 (예: ReLU, Sigmoid, Softmax 등).

---

## 4. 순전파 및 역전파 (Forward & Backward Pass)

* **순전파 (Feed Forward / Forward Pass):** 입력 데이터가 `입력층` $\rightarrow$ `은닉층` $\rightarrow$ `출력층` 방향으로 이동하며 최종 예측값($\hat{y}$)을 계산하는 과정.
* **역전파 (Backward Pass / Backpropagation):** 예측 오차(Loss)를 줄이기 위해, 손실을 출력층에서 입력층 반대 방향으로 거꾸로 전달하면서 각 가중치($w$)를 얼마큼 수정해야 할지 계산하는 알고리즘.
* **체인 룰 / 연쇄 법칙 (Chain Rule):** 역전파 연산 시 미분의 연쇄적인 성질을 이용해 미분값(기울기)을 순차적으로 구해내는 수학적 원리.

---

## 5. 성능 평가, 오차 및 지표 (Loss & Metric)

* **예측값 (Prediction, $\hat{y}$):** 모델이 입력 데이터($x$)를 받아 예측해 낸 결과값.
* **오차 (Error):** 실제 정답($y$)과 모델 예측값($\hat{y}$)의 차이 ($\hat{y} - y$).
* **손실 함수 (Loss Function):** 모델 학습 시 오차의 크기를 측정하는 컴퓨터 최적화용 수학 공식 (작을수록 좋으며, 미분 가능해야 함).
  * **MSE (Mean Squared Error, 평균 제곱 오차):** 예측값과 실제값 차이를 제곱하여 평균을 낸 회귀 기본 손실 함수.
  * **Cross-Entropy:** 분류 문제에서 예측 확률 분포와 실제 정답 간의 차이를 계산하는 손실 함수.
* **평가지표 (Metric):** 학습 완료 후 사람이 직관적으로 성능을 판단하기 위한 평가 기준 (예: Accuracy, Precision, Recall).
  * **RMSE (Root Mean Squared Error):** MSE에 루트를 씌워 실제 데이터와 동일한 단위 스케일로 정렬한 평가 지표.
  * **MAE (Mean Absolute Error):** 오차의 절댓값을 평균 낸 지표로, 이상치(Outlier)의 영향에 상대적으로 강함.
  * **정확도 (Accuracy):** 전체 예측 데이터 중 정답을 맞춘 데이터 비율.

---

## 6. 학습 상태 및 최적화 (Optimization & Regularization)

* **과적합 (Overfitting):** 모델이 훈련 데이터의 잡음(Noise)까지 너무 과도하게 외워서, 새로운 테스트 데이터에서는 성능이 떨어지는 현상.
* **과소적합 (Underfitting):** 모델의 표현력이 부족하거나 학습이 덜 되어, 훈련 데이터조차 충분히 학습하지 못하는 현상.
* **최적화 (Optimization):** Loss 수치를 최소화할 수 있도록 모델의 파라미터(가중치 $w$)를 조절해 나가는 과정.
* **기울기 / 그라디언트 (Gradient):** 손실 함수 그래프에서 Loss가 가장 가파르게 변하는 방향과 변화율.
* **학습률 (Learning Rate, LR):** 최적화 과정에서 가중치를 한 번 수정할 때 이동하는 수치 범위(발걸음의 폭).
* **경사 하강법 (Gradient Descent):** 기울기를 계산해 그 반대 방향으로 조금씩 가중치를 이동시켜 손실(Loss)의 최솟값을 찾아가는 최적화 알고리즘.
* **확률적 경사 하강법 (SGD):** 무작위로 뽑은 미니 배치(Mini-batch) 데이터를 이용하여 빠르게 가중치를 업데이트하는 경사 하강법.
* **옵티마이저 (Optimizer):** 경사 하강법을 발전시켜 효율적으로 가중치를 업데이트하는 최적화 도구 (예: SGD, Momentum, Adam, RMSProp 등).
* **규제 (Regularization):** 과적합을 방지하기 위해 가중치의 크기에 패널티를 부여하여 과도한 학습을 억제하는 기법 (예: L1 Lasso, L2 Ridge, Dropout 등).
* **배치 정규화 (Batch Normalization):** 레이어 사이에서 미니 배치 단위로 데이터의 분포를 정돈하여 학습을 빠르고 안정적으로 도와주는 기술.
* **조기 종료 (Early Stopping):** 검증 손실(Validation Loss)이 더 이상 감소하지 않고 다시 증가하기 시작하는 시점(과적합 시작점)에서 학습을 자동으로 중단하는 기술.
