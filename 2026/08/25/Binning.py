import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 💡 표를 터미널 창에 맞춰서 예쁘게 정렬해 주는 설정입니다.
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)

salary = pd.Series([4300, 8370, 1750, 3830, 1840, 4220, 3020, 2290, 4740, 4600,
                    2860, 3400, 4800, 4470, 2440, 4530, 4850, 4850, 4760, 4500,
                    4640, 3000, 1880, 4880, 2240, 4750, 2750, 2810, 3100, 4290,
                    1540, 2870, 1780, 4670, 4150, 2010, 3580, 1610, 2930, 4300,
                    2740, 1680, 3490, 4350, 1680, 6420, 8740, 8980, 9080, 3990,
                    4960, 3700, 9600, 9330, 5600, 4100, 1770, 8280, 3120, 1950,
                    4210, 2020, 3820, 3170, 6330, 2570, 6940, 8610, 5060, 6370,
                    9080, 3760, 8060, 2500, 4660, 1770, 9220, 3380, 2490, 3450,
                    1960, 7210, 5810, 9450, 8910, 3470, 7350, 8410, 7520, 9610,
                    5150, 2630, 5610, 2750, 7050, 3350, 9450, 7140, 4170, 3090])
# salary.hist()
# plt.show()


bins = [0, 2000, 4000, 6000, 8000, 10000]


#bins = 몇 등분 할거야?
ctg = pd.cut(salary, bins=bins)
print(ctg)
print('salary[0]:', salary[0])
print('salary[0]가 속한 카테고리:', ctg[0])
print(ctg.value_counts().sort_index())

# qcut 에 q = 몇 % 로 자를거야?
ctg = pd.qcut(salary, q=5)
print(ctg)