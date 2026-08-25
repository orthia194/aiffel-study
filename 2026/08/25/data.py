import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 💡 표를 터미널 창에 맞춰서 예쁘게 정렬해 주는 설정입니다.
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)

csv_file_path = 'd:/vscode/aiffel/2026/08/25/data/trade.csv'
trade = pd.read_csv(csv_file_path)
trade.head()

# print(trade.head())

# print('전체 데이터 건수:', len(trade))

# print('컬럼별 결측치 개수')
# print(len(trade) - trade.count())

#기타사항 컬럼 삭제
trade = trade.drop('기타사항', axis=1)
# print(trade.head())

# print(trade.isnull())
# print(trade.isnull().any(axis=1))
# print(trade[trade.isnull().any(axis=1)])

trade.dropna(how='all', subset=['수출건수', '수출금액', '수입건수', '수입금액', '무역수지'], inplace=True)
# print("👽 It's okay, no biggie.")

# print(trade[trade.isnull().any(axis=1)])

# print(trade[(trade['국가명']=='미국')&((trade['기간']=='2020년 03월')|(trade['기간']=='2020년 05월'))])

trade.loc[191, '수출금액'] = (trade.loc[188, '수출금액'] + trade.loc[194, '수출금액'] )/2

# print(trade.loc[[191]])

trade.loc[191, '무역수지'] = (trade.loc[191, '수출금액'] - trade.loc[191, '수입금액'] )
# print(trade.loc[[191]])

# print(len(trade) - trade.count())


# print(trade.head())

# print(trade.duplicated())
# print(trade.duplicated()[trade.duplicated()])

# print(trade[trade.duplicated()])

# print(trade[(trade['기간']=='2020년 03월')&(trade['국가명']=='중국')])

trade.drop_duplicates(inplace=True)

# print(trade.duplicated()[trade.duplicated()])

def outlier(df, col, z):
    return df[abs(df[col] - np.mean(df[col]))/np.std(df[col])>z].index

# print(trade.loc[outlier(trade, '무역수지', 1.5)],'\n')
# print(trade.loc[outlier(trade, '무역수지', 2)],'\n')
# print(trade.loc[outlier(trade, '무역수지', 3)],'\n')

def not_outlier(df, col, z):
    return df[abs(df[col] - np.mean(df[col]))/np.std(df[col])<=z].index

# print(trade.loc[not_outlier(trade, '무역수지', 1.5)],'\n')

#------------------------------------------------------------------------------------

def outlier2(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > 1.5 * IQR)]

# print(outlier2(trade, '무역수지'))

def outlier3(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

# print(outlier3(trade, '무역수지'))

## 이상치가 나오지 않았다면 데이터가 아주 깔끔하고 안정적으로 잘 정제되어있다 라고 봐도 무방하다. - 제미나이
## 데이터 수치만으로 이해하기 힘들 땐 그래프나 그림으로 확인하는게 가장 좋다.
#------------------------------------------------------------------------------------

# trade 데이터를 standardization 기법으로 정규화합니다.
cols = ['수출건수', '수출금액', '수입건수', '수입금액', '무역수지']
trade_standardization = (trade[cols]-trade[cols].mean())/trade[cols].std()
print(trade_standardization.head(),'\n')

print(trade_standardization.describe(),'\n')


# Q. trade 데이터를 min-max scaling 기법으로 정규화합니다.
for col in cols:
    min_val = trade[col].min()
    max_val = trade[col].max()
    trade[col] = (trade[col] - min_val) / (max_val - min_val)
print(trade.head(),'\n')

print(trade.describe(),'\n')

#------------------------------------------------------------------------------------

#trade 데이터의 국가명 컬럼 원본
print(trade['국가명'].head(),'\n')

# get_dummies를 통해 국가명 원-핫 인코딩
country = pd.get_dummies(trade['국가명'])
print(country.head(),'\n')

trade = pd.concat([trade, country], axis=1)
print(trade.head())