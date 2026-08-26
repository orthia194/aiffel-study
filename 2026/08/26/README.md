# 📌 머신러닝 평가 지표 핵심 정리

## 1. Loss vs Metric

* **Loss (손실 함수)**
  * **목적:** 모델 학습(최적화)용 수학적 수치
  * **특징:** 알고리즘이 가중치(Weight)를 업데이트하기 위해 사용되며, **미분 가능**해야 하고 **작을수록** 좋습니다.
  * **예시:** MSE, Cross-Entropy

* **Metric (평가 지표)**
  * **목적:** 사람을 위한 모델 성능 평가용 수치
  * **특징:** 모델 학습 완료 후 사람이 직관적으로 성능을 판단하기 위한 기준입니다.
  * **예시:** Accuracy, Precision, Recall, F1-Score

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
