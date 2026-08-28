### ⚖️ Regularization vs Normalization

한국어로 둘 다 '정규화'로 번역되는 경우가 많아 헷갈리기 쉬운 **Regularization**과 **Normalization**의 핵심 차이점 비교입니다.

| 구분 | Regularization (규제 / 정칙화) | Normalization (정규화) |
| :--- | :--- | :--- |
| **주요 목적** | **과적합(Overfitting) 방지** 및 모델 일반화 성능 향상 | **데이터 스케일(범위) 통일** 및 학습 안정화·속도 향상 |
| **적용 대상** | **모델의 가중치($W$)** 또는 모델 구조 자체 | **입력 데이터(Features)** 또는 **레이어의 활성화값(Activation)** |
| **핵심 원리** | 가중치($W$)가 너무 커지지 않도록 손실함수에 패널티를 부여하거나 노드를 생략함 | 데이터 분포의 범위를 일정한 구간(예: $0 \sim 1$ 또는 평균 0, 표준편차 1)으로 맞춤 |
| **주요 기법** | L1 규제 (Lasso), L2 규제 (Ridge), Dropout, Early Stopping | Min-Max Scaling, Standard Scaling, Batch Normalization |
| **대표 비유** | 모델이 너무 암기하지 못하도록 **족쇄/제약**을 채우는 것 | 선수들의 **체급(스케일)**을 공평하게 맞춰주는 것 |


**regularization의 목적은 처음 보는 데이터(validation/test)에서도 잘 맞히는 것**  
train loss는 약간 올라갈 수 있지만, 대신 validation/test loss가 낮아지는 것을 노립니다.

---

💡 **참고 지식: 과적합(Overfitting)은 왜/언제 생기는가?**

과적합은 모델이 데이터의 **일반적인 패턴**을 학습하는 것이 아니라, 훈련 데이터의 **사소한 특징(노이즈)까지 통째로 외워버릴 때** 발생합니다.

---

### **1. 과적합이 주로 발생하는 4가지 상황**

* 📉 **데이터 문제:** 학습 데이터가 부족하거나 품질이 낮을 때
* ⚙️ **모델 구조 문제:** 모델이 데이터에 비해 너무 복잡할 때 *(파라미터 수 과다)*
* ⏳ **학습량 문제:** 같은 데이터를 필요 이상으로 너무 오래 학습할 때 *(Epoch 과다)*
* 🧹 **피처 문제:** 불확실하거나 노이즈(잡음)가 많은 Feature가 섞여 있을 때

---

### **2. 과적합 진단 방법 (Learning Curve)**

과적합 여부를 확인하는 가장 대표적인 방법은 **Train Loss**와 **Validation Loss**를 하나의 그래프에 그려 비교해 보는 것입니다.

* **정상 학습:** Train Loss와 Validation Loss가 함께 원만하게 감소함
* **과적합 발생:** Train Loss는 계속 줄어들지만, Validation Loss가 어느 시점부터 정체되거나 다시 상승함 (두 곡선의 격차가 벌어짐)

---

### 🔄 Epoch, Batch Size, Iteration 개념 비교

| 개념 | 영문명 | 핵심 의미 | 쉬운 비유 (100페이지 문제집 기준) |
| :--- | :--- | :--- | :--- |
| **에포크** | **Epoch** | 전체 데이터를 처음부터 끝까지 **1회 학습**하는 단위 | 문제집 100페이지를 **1회독 완료** |
| **배치 크기** | **Batch Size** | 한 번의 묶음 학습에 사용하는 **데이터 개수** | 한 번에 풀 문제 분량 (**10페이지씩** 묶음) |
| **이러레이션 / 스텝** | **Iteration / Step** | 1 Epoch를 마치기 위해 **배치를 제출/학습한 횟수** | 100페이지를 10페이지씩 풀었을 때 **총 제출 횟수 (10회)** |

> 💡 **수식 관계:**  
> $\text{1 Epoch} = \text{Batch Size} \times \text{Iteration (Steps)}$

# 🎯 머신러닝 규제(Regularization): L1 (Lasso) vs L2 (Ridge)

선형 회귀(Linear Regression) 모델의 과대적합(Overfitting)을 방지하고 일반화 성능을 높이기 위해 사용하는 **L1 규제(Lasso)**와 **L2 규제(Ridge)**의 개념, 차이점, 사용 시기를 정리한 가이드입니다.

---

## 📌 1. 규제(Regularization)란 왜 필요할까?

기본 **선형 회귀(Linear Regression)**는 주어진 데이터의 오차(Loss)만을 최소화하는 방향으로 학습합니다. 
그러나 데이터에 노이즈가 많거나 특징(Feature/열)의 개수가 너무 많으면, 모델이 학습 데이터에 지나치게 맞춰져 **과대적합(Overfitting)**이 발생합니다.

* **손실 함수 (Basic Loss Function):**
  $$\text{Loss} = \text{MSE (Mean Squared Error)}$$

규제(Regularization)는 손실 함수에 **가중치(Weight/기울기)에 대한 패널티(Penalty) 항**을 추가하여 가중치가 너무 커지지 않도록 억제하는 기법입니다.

---

## ⚖️ 2. L1 Regularization (Lasso) vs L2 Regularization (Ridge)

| 구분 | **L1 Regularization (Lasso)** | **L2 Regularization (Ridge)** |
| :--- | :--- | :--- |
| **수학적 패널티** | 가중치 **절댓값의 합** ($\vert{}w\vert{}$) | 가중치 **제곱의 합** ($w^2$) |
| **손실 함수** | $\text{MSE} + \alpha \sum \vert{}w_i\vert{}$ | $\text{MSE} + \alpha \sum w_i^2$ |
| **가중치 변화** | 불필요한 특징의 가중치를 **정확히 0**으로 만듦 | 가중치를 **0에 가깝게 완만하게 줄임** (0은 안 됨) |
| **주요 효과** | **특징 선택 (Feature Selection)** / 모델 단순화 | **과대적합 방지** / 가중치 안정화 |
| **주요 사용 상황** | **특징(열/Column)의 개수가 매우 많을 때** | 모든 특징이 어느 정도 의미를 가질 때 |
| **주의할 점** | $\alpha$가 너무 크면 중요 변수도 0이 되어 **가로선($y=b$)**이 됨 | 특징 개수를 줄이지 않으므로 메모리/계산 유지 |

---

## 🔍 3. 심층 분석: 작동 원리 및 특징

### 🥊 L1 규제 (Lasso - Least Absolute Shrinkage and Selection Operator)
> **"필요 없는 열(Column)은 싹둑 잘라내어 지워버리는 칼"**

* **핵심 특징:** 
  * 중요도가 낮은 특징의 가중치(기울기)를 **완전히 `0`**으로 만듭니다.
  * 100개의 특징 중 유의미한 10개만 남기고 나머지는 제거하는 **Feature Selection** 효과가 있습니다.
* **가로선 문제 발생 원인:**
  * 만약 특징이 단 1개만 있는 상태에서 L1 규제 강도($\alpha$)를 너무 높이면, 하나뿐인 특징의 기울기마저 `0`이 되어 $y = 0 \cdot x + b \Rightarrow y = b$ 형태의 **평평한 가로선**을 그려버립니다.

### 🛡️ L2 규제 (Ridge)
> **"모든 특징을 살려두되, 영향력을 부드럽게 낮추는 정지 마찰력"**

* **핵심 특징:**
  * 모든 가중치를 **0에 가깝게 균일하게 축소**하지만, 절대 **`0`으로 만들지는 않습니다.**
  * 다중공선성(Multi-collinearity, 특징들 간의 강한 상관관계)이 존재할 때 매우 안정적인 성능을 보입니다.
  * 그래프상에서 대각선의 기울기가 완만해질 뿐, 데이터의 전반적인 경향선(대각선)을 계속 유지합니다.

---

## 💻 4. Python (Scikit-Learn) 실습 코드

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# 1. 데이터 준비 (Pandas [[ ]]를 사용하여 처음부터 2차원 DataFrame으로 추출)
X = iris_df.loc[iris_df['species'] == 'virginica', ['petal length (cm)']] # Shape: (50, 1) - 2D
Y = iris_df.loc[iris_df['species'] == 'virginica', 'sepal length (cm)']   # Shape: (50,) - 1D

# 2. 기본 선형 회귀 (Linear Regression)
linear = LinearRegression()
linear.fit(X, Y)

# 3. L1 규제 회귀 (Lasso) - alpha 조절 필수
lasso = Lasso(alpha=0.01) # alpha가 너무 크면 기울기가 0이 되어 가로선이 됨
lasso.fit(X, Y)

# 4. L2 규제 회귀 (Ridge)
ridge = Ridge(alpha=1.0)
ridge.fit(X, Y)

print(f"Linear 기울기: {linear.coef_[0]:.4f}")
print(f"Lasso  기울기: {lasso.coef_[0]:.4f}")
print(f"Ridge  기울기: {ridge.coef_[0]:.4f}")
