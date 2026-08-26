import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 💡 표를 터미널 창에 맞춰서 예쁘게 정렬해 주는 설정입니다.
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)

csv_path = 'd:/vscode/aiffel/2026/08/25/data/pokemon.csv'
original_data = pd.read_csv(csv_path)

pokemon = original_data.copy()  # 1. 원본 안전하게 복사
print(pokemon.shape)            # 2. 전체 크기(행, 열 개수) 확인
pokemon.head()                  # 3. 상단 데이터가 어떻게 생겼나 눈으로 확인