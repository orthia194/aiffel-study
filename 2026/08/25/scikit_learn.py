import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# print(sklearn.__version__)

r = np.random.RandomState(10)
x = 10 * r.rand(100)
y = 2 * x - 3 * r.rand(100)
plt.scatter(x,y)
# plt.show()

print(x.shape)
print(y.shape,'\n')

from sklearn.linear_model import LinearRegression
model = LinearRegression()
# print(model)

# X는 2차원 형태가 필수 이기 때문에 (100,1) 로 변경해줘야 함
# x.reshape(-1, 1) - 데이터가 100개든 1,000개든 개수가 바뀌어도 코드를 수정할 필요가 없어서 실무에서 훨씬 자주 쓰입니다.
X = x.reshape(100,1)
print(model.fit(X,y))

# 새로운 데이터 준비 및 예측
x_new = np.linspace(-1, 11, 100)
X_new = x_new.reshape(100, 1)
y_new = model.predict(X_new)

# 예측 결과 확인
print('예측결과\n',y_new,'\n')

X_ = x_new.reshape(-1,1)
print(X_.shape)

error = np.sqrt(mean_squared_error(y,y_new))

print(error)

plt.scatter(x, y, label='input data')
plt.plot(X_new, y_new, color='red', label='regression line')
# plt.show()