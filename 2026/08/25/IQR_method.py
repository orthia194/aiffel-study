import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(2020)
data = np.random.randn(100)  # 평균 0, 표준편차 1의 분포에서 100개의 숫자를 샘플링한 데이터 생성
data = np.concatenate((data, np.array([8, 10, -3, -5])))      # [8, 10, -3, -5])를 데이터 뒤에 추가함

print(data)

#그림 그리기
# fig, ax = plt.subplots()
# ax.boxplot(data)
# plt.show()

# IQR = Q3 - Q1 의 값
# 이상치 = IQR 의 값에서 1.5배를 한 후 Q1 - IQR , Q3 + IQR 를 한 값
# if 
# Q1 = 20
# Q3 = 30
# IQR = 30 - 20 = 10
# 이상치 = IQR x 1.5(15)
# Q1 = 5
# Q3 = 45

Q3, Q1 = np.percentile(data, [75 ,25])
IQR = Q3 - Q1
print(IQR)

print(data[(Q1-1.5*IQR > data)|(Q3+1.5*IQR < data)])