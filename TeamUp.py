"""
Created on Wed Mar 28 22:29:39 2018
@author: TeamUp
"""
# 이제 여기에다 코드를 써내려(복붙) 가야지

#%%(1)

import pandas as pd
from pandas import Series, DataFrame

art = pd.read_csv('/Users/hbk/github/TeamUp/크롤링 데이터셋/art.csv')
art
art.columns = ['idx','제목','삭제','내용']
art.head()
art = art[['제목','내용']]
art.head()

from konlpy.tag import Twitter
t = Twitter()

res = {}
num = 0
for i in art['제목'].values:
    txt = t.pos(i)
    lst = []
    for j in txt:
        if j[1] == 'Noun':
            lst.append(j[0])
    res[num] = lst
    num += 1
        
res  # dict
df = DataFrame(Series(res))
df.columns = ['art']
df
