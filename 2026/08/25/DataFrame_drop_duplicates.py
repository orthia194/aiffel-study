import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 💡 표를 터미널 창에 맞춰서 예쁘게 정렬해 주는 설정입니다.
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)

df = pd.DataFrame({'id':['001', '002', '003', '004', '002'],
                   'name':['Park Yun', 'Kim Sung', 'Park Jin', 'Lee Han', 'Kim Min']})
print("처음 값\n",df)

# 중복 데이터 제거 id값의 제일 나중 값만 살리고 나머지는 삭제
# ignore_index = 재정렬 해주는 옵션
df.drop_duplicates(subset=['id'], keep='last', inplace=True , ignore_index=True)
print("변경 값\n",df)