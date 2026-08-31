# 📌 머신러닝 평가 지표 핵심 정리

## 1. Loss vs Metric

| 구분 | **Loss (손실 함수)** | **Metric (평가 지표)** |
| :--- | :--- | :--- |
| **주요 대상** | **컴퓨터(알고리즘)**를 위한 수치 | **사람(개발자/사용자)**을 위한 수치 |
| **핵심 목적** | **모델 학습(최적화)** 및 가중치(Weight) 업데이트 | 모델의 **최종 성능 평가** 및 비즈니스 판단 |
| **주요 특징** | • **미분 가능(Differentiable)**해야 함 (경사하강법 적용)<br>• 연속적인 변화량을 가짐<br>• 항상 **작을수록 좋음** | • 미분 불가능해도 상관없음 (계단형/불연속 수치 가능)<br>• 직관적이며 해석하기 쉬움<br>• 지표에 따라 **높을수록 좋음** (예: 정확도) |
| **대표 예시** | MSE, MAE, Cross-Entropy, Huber Loss | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

---

### 💡 왜 Loss와 Metric을 구분할까요?

1. **미분 가능성 (Differentiable)**
   * 모델이 학습(가중치 업데이트)을 하려면 **경사하강법(Gradient Descent)**을 사용해 기울기(미분값)를 구해야 합니다.
   * `Accuracy(정확도)` 같은 지표는 "맞았다(1) / 틀렸다(0)"처럼 불연속적이라 미분이 불가능하므로, 학습용으로는 미분이 가능한 **Loss(예: Cross-Entropy)**를 사용합니다.

2. **비즈니스 목적성**
   * 손실 함수(Loss)는 모델을 수학적으로 줄여나가는 내부 도구일 뿐입니다.
   * 실제 서비스에서는 오차(Loss) 수치보다 **"암 환자를 얼마나 잘 찾아냈는가(Recall)"**처럼 사람이 직관적으로 이해할 수 있는 **Metric**이 평가 기준이 됩니다.

---

## 2. Confusion Matrix (혼동 행렬)

| | **예측: Positive (양성)** | **예측: Negative (음성)** |
|---|---|---|
| **실제: Positive** | **TP** (True Positive)<br>실제 양성을 양성으로 맞춤 | **FN** (False Negative)<br>실제 양성을 음성으로 틀림 |
| **실제: Negative** | **FP** (False Positive)<br>실제 음성을 양성으로 틀림 | **TN** (True Negative)<br>실제 음성을 음성으로 맞춤 |

분류 모델의 예측 결과를 정답과 비교하여 4가지로 분류한 표입니다.
쉽게 이해하려면 **"Positive = 범인(우리가 찾으려는 대상)"**, **"Negative = 일반 시민(정상)"**으로 비유하면 쉽습니다.

| | **예측: Positive (범인으로 지목)** | **예측: Negative (시민으로 판정)** |
|---|---|---|
| **실제: Positive (진짜 범인)** | **TP** (True Positive)<br>범인을 정확히 체포함 | **FN** (False Negative)<br>범인을 못 알아보고 놓침 *(2종 오류)* |
| **실제: Negative (일반 시민)** | **FP** (False Positive)<br>시민에게 억울하게 누명을 씌움 *(1종 오류)* | **TN** (True Negative)<br>시민을 정상적으로 돌려보냄 |

### 💡 용어 쉽게 읽는 법 (2글자 법칙)
* **뒤쪽 글자 (Positive / Negative):** 모델이 **무엇으로 예측했는지** (범인으로 봤나? 시민으로 봤나?)
* **앞쪽 글자 (True / False):** 그 예측이 **맞았는지 / 틀렸는지**
---

## 3. Precision vs Recall

### 🎯 Precision (정밀도)
* **공식:** $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
* **개념:** 모델이 **양성이라고 예측한 것** 중 진짜 양성의 비율
* **주요 사용처:** **음성을 양성으로 잘못 예측하면 치명적인 경우**
  * *예시:* 스팸 메일 분류 (일반 중요한 메일을 스팸으로 분류하면 안 됨)

### 🔍 Recall (재현율 / 민감도)
* **공식:** $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
* **개념:** **실제 양성인 데이터** 중 모델이 양성으로 맞춘 비율
* **주요 사용처:** **양성을 놓치면 치명적인 경우**
  * *예시:* 암 환자 진단, 금융 사기(FDS) 감지 (암 환자를 정상으로 오진하면 안 됨)

---

## 4. Threshold (임계값) 변화에 따른 성능 변화

모델이 출력한 **양성 확률을 분류하는 기준선** (기본값: 0.5)

* **Threshold 낮춤 (0.5 → 0.3)**
  * 기준이 관대해져서 웬만하면 양성으로 판정
  * **Recall 상승 ↑ / Precision 하강 ↓**

* **Threshold 높임 (0.5 → 0.7)**
  * 기준이 깐깐해져서 확실한 경우만 양성으로 판정
  * **Precision 상승 ↑ / Recall 하강 ↓**

> ⚠️ **Trade-off (시소 관계):** Precision과 Recall은 한쪽이 올라가면 다른 쪽이 떨어지는 반비례 관계입니다.
>
> ## 📊 5. Threshold(임계값) 변경을 통한 성능 조절 및 PR 커브

### 1. 임계값(Threshold)이란?
* `decision_function()`이 출력하는 **`y_score`**(결정 경계로부터의 거리/확신도 점수)를 기준으로 `0`과 `1`을 분류하는 **판단 기준선**입니다.
* 기본값(Default)은 `0`이지만, 분석 목적에 따라 기준을 높이거나 낮출 수 있습니다.

# 임계값을 기본값(0)에서 -0.2로 낮추어 1(Positive) 판정을 더 넓게 적용
y_pred_new = classifier.decision_function(X_test) > -0.2

# 새 임계값 기준 평가 리포트 확인
print(confusion_matrix(y_test, y_pred_new))
print(classification_report(y_test, y_pred_new))

# 📊 정밀도(Precision), 재현율(Recall) 및 PR 커브 정리

## 1. 개념 정의

* **정밀도 (Precision)**: 모델이 `1`(Positive)이라고 예측한 것 중 **실제 `1`인 비율**
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  * *의미:* 모델이 맞다고 한 것 중에서 "진짜 맞춘 비율" (오탐/허위 경보 방지)
* **재현율 (Recall)**: 실제 `1`(Positive)인 전체 데이터 중 **모델이 놓치지 않고 찾아낸 비율**
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  * *의미:* 원래 있던 진짜 `1` 중에서 "몇 개나 긁어모았는가" (누락 방지)

---
![Precision and Recall Diagram](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)

## 2. 임계값(Threshold)과 트레이드오프 (Trade-off)

`decision_function()`이 반환하는 확신 점수(`y_score`)의 판단 기준선(Threshold)을 조절하면, **정밀도와 재현율은 시소(Trade-off)처럼 반대로 작동**합니다.

from sklearn.metrics import classification_report, confusion_matrix

# 임계값을 기본값(0)에서 -0.2로 낮추어 positive 판정 기준 완화
y_pred_new = classifier.decision_function(X_test) > -0.2

# 새 임계값 기준 평가 채점표 출력
print(confusion_matrix(y_test, y_pred_new))
print(classification_report(y_test, y_pred_new))

| 구분 | 임계값 (Threshold) 조절 | Precision (정밀도) | Recall (재현율) | 대표 활용 분야 |
| :--- | :--- | :---: | :---: | :--- |
| **높임** | `> +0.2` (확실한 것만 1로 분류) | **▲ 상승** | **▼ 하락** | 스팸 메일 분류 (정상 메일 차단 방지) |
| **낮춤** | `> -0.2` (의심스러운 것도 1로 분류) | **▼ 하락** | **▲ 상승** | 암 진단, 금융 사기 탐지 (위험 요소 누락 방지) |
