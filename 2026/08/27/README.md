### ⚖️ Regularization vs Normalization

한국어로 둘 다 '정규화'로 번역되는 경우가 많아 헷갈리기 쉬운 **Regularization**과 **Normalization**의 핵심 차이점 비교입니다.

| 구분 | Regularization (규제 / 정칙화) | Normalization (정규화) |
| :--- | :--- | :--- |
| **주요 목적** | **과적합(Overfitting) 방지** 및 모델 일반화 성능 향상 | **데이터 스케일(범위) 통일** 및 학습 안정화·속도 향상 |
| **적용 대상** | **모델의 가중치($W$)** 또는 모델 구조 자체 | **입력 데이터(Features)** 또는 **레이어의 활성화값(Activation)** |
| **핵심 원리** | 가중치($W$)가 너무 커지지 않도록 손실함수에 패널티를 부여하거나 노드를 생략함 | 데이터 분포의 범위를 일정한 구간(예: $0 \sim 1$ 또는 평균 0, 표준편차 1)으로 맞춤 |
| **주요 기법** | L1 규제 (Lasso), L2 규제 (Ridge), Dropout, Early Stopping | Min-Max Scaling, Standard Scaling, Batch Normalization |
| **대표 비유** | 모델이 너무 암기하지 못하도록 **족쇄/제약**을 채우는 것 | 선수들의 **체급(스케일)**을 공평하게 맞춰주는 것 |
