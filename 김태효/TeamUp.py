"""
Created on Wed Mar 28 22:29:39 2018
@author: TeamUp
"""
# 이제 여기에다 코드를 써내려(복붙) 가야지


import pandas as pd
from pandas import Series, DataFrame

art = pd.read_csv('/Users/hbk/github/TeamUp/크롤링 데이터셋/art.csv')
art
art.columns = ['idx','제목','삭제','내용']
art = art[['제목','내용']]
art.head()

from konlpy.tag import Twitter
t = Twitter()

res = {}
num = 0
df = DataFrame()

for i in art['제목'].values:
    #txt = t.pos(i)
    txt = t.morphs(i)
    res[num] = txt
    num += 1
res
df = DataFrame(Series(res), columns = ['art'])
df


import numpy as np

def cosineDist(a,b):
    inner_ab = a.dot(b)
    norm_a = np.sqrt(a.dot(a))
    norm_b = np.sqrt(b.dot(b))
    
    return inner_ab/(norm_a*norm_b)

a = np.array([1,0])
b = np.array([2,3])

cosineDist(a,b)

=====
# 사용된 모든 형태소 목록만드는 클래스?
import numpy as np

class Vocab:
    def __init__(self):
        self.vector = {}
    
    def add(self,tokens):
        for token in tokens:
            if token not in self.vector and not token.isspace() and token != '':
                self.vector[token] = len(self.vector)
    
    def indexOf(self,vocab):
        return self.vector[vocab]
    
    def size(self):
        return len(self.vector)
    
    def at(self,i): # get ith word in the vector
        return list(self.vector)[i]
    
    # vectorize : dict -> np.array
    
    def vectorize(self,word):
        v = [0 for i in range(self.size())]
        if word in self.vector:
            v[self.indexOf(word)]=1
        else:
            print("<ERROR>Word\'"+word+"\'Not Found")
        return np.array(v)
    
    def save(self,filename):
        with open(filename,'w',encoding='utf-8') as f:
            for word in self.vector:
                f.write(word+'\n')
                
    def load(self,filename):
        with open(filename,'r',encoding='utf-8') as f:
            lines = f.readlines()
            bow = [i[:-1] for i in lines]
            self.add(bow)
    
    def __str__(self):
        s = "Vocab("
        for word in self.vector:
            s += (str(self.vector[word]) + ':' + word + ',')
        if self.size() != 0:
            s = s[:-2]
        s += ")"
        return s
import nltk        
from nltk.corpus import stopwords

# str -> nltk.Text로 리턴하는 함수

def preprocess(x):
    #x = x.lower()
    tokens = nltk.word_tokenize(x)
    stop = set(stopwords.words('english'))
    tokens = [i for i in tokens if i not in stop and i.isalpha()]
    stemmer = nltk.stem.porter.PorterStemmer()
    stems = [stemmer.stem(i) for i in tokens]
    text = nltk.Text(stems)
    return text


nltk.download('punkt')
for i in art['제목'].values:
    print(preprocess(i))







