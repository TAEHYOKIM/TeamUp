import urllib.request as req
from bs4 import BeautifulSoup
import os
from os.path import exists
import re
import time
import csv
from pandas import Series,DataFrame
import pandas as pd
import numpy as np

#edu
z = np.zeros((1980,3))
edu = DataFrame(z)

cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=11&queryTime=2018-03-29%2016%3A49%3A41&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        edu[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + edu[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            #print(len(k.get_text(strip = True)))
            edu[2][h + cnt_id] = k.get_text(strip = True)
        
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            edu[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)



################################################################################
edu.to_csv("c:/python/edu.csv", mode='w',encoding = 'utf-8')
################################################################################



#computer
z = np.zeros((1980,3))
computer = DataFrame(z)

cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=1&queryTime=2018-03-29%2016%3A50%3A03&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        computer[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + computer[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            computer[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            computer[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
computer.to_csv("c:/python/computer.csv", mode='w',encoding = 'utf-8')
################################################################################

#game
z = np.zeros((1980,3))
game = DataFrame(z)

cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=2&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        game[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + game[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            game[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            game[0][h + cnt_id] = f.get_text(strip = True)            
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
game.to_csv("c:/python/game.csv", mode='w',encoding = 'utf-8')
################################################################################


#art
z = np.zeros((1980,3))
art = DataFrame(z)

cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=3&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        art[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + art[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            art[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            art[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
art.to_csv("c:/python/art.csv", mode='w',encoding = 'utf-8')
################################################################################



#living
z = np.zeros((1980,3))
living = DataFrame(z)

cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=8&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")

    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        living[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + living[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            living[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            living[0][h + cnt_id] = f.get_text(strip = True)            
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
living.to_csv("c:/python/living.csv", mode='w',encoding = 'utf-8')
################################################################################



#health
z = np.zeros((1980,3))
health = DataFrame(z)

cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=7&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        health[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + health[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            health[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            health[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
health.to_csv("c:/python/health.csv", mode='w',encoding = 'utf-8')
################################################################################




#soc
z = np.zeros((1980,3))
soc = DataFrame(z)

cnt_title = 0
cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=6&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        soc[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + soc[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            soc[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            soc[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)

################################################################################
soc.to_csv("c:/python/soc.csv", mode='w',encoding = 'utf-8')
################################################################################





#eco
z = np.zeros((1980,3))
eco = DataFrame(z)

cnt_title = 0
cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=4&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        eco[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + eco[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            eco[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            eco[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
eco.to_csv("c:/python/eco.csv", mode='w',encoding = 'utf-8')
################################################################################




#travel
z = np.zeros((1980,3))
travel = DataFrame(z)

cnt_title = 0
cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=9&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        travel[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + travel[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            travel[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            travel[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
travel.to_csv("c:/python/travel.csv", mode='w',encoding = 'utf-8')
################################################################################





#sports
z = np.zeros((1980,3))
sports = DataFrame(z)

cnt_title = 0
cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=10&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        sports[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + sports[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            sports[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            sports[0][h + cnt_id] = f.get_text(strip = True)
            
    cnt_id += 20
    print(cnt_id)
    

################################################################################
sports.to_csv("c:/python/sports.csv", mode='w',encoding = 'utf-8')
################################################################################




#shopping
z = np.zeros((1980,3))
shopping = DataFrame(z)

cnt_title = 0
cnt_id = 0

for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=5&queryTime=2018-03-29%2016%3A50%3A38&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
  
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        shopping[1][cnt_id+j] = soup3
        
    for h in range(0,20):
        url = "http://kin.naver.com" + shopping[1][h+cnt_id]
        res = req.urlopen(url)
        soup = BeautifulSoup(res,"html.parser")
            
        l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
        for k in l:
            shopping[2][h + cnt_id] = k.get_text(strip = True)
            
        l = soup.select("#qna_detail_question > div.end_question._end_wrap_box > div.end_tit > div.tit_cont._endTitle > div > h3 > span.title_text")
        
        for f in l:
            shopping[0][h + cnt_id] = f.get_text(strip = True)

    cnt_id += 20
    print(cnt_id)
    

################################################################################
shopping.to_csv("c:/python/shopping.csv", mode='w',encoding = 'utf-8')
################################################################################



"""
    
shopping_title = []
shopping_text = []
for i in range(1,100):
    url = "http://kin.naver.com/qna/list.nhn?dirId=11&queryTime=2018-03-28%2014%3A12%3A17&page="+str(i)
    res = req.urlopen(url)
    soup = BeautifulSoup(res,"html.parser")
    
    l = soup.select("#au_board_list > tr > td.title > a")

    for i in l:
        shopping_title.append(i.string)
        
    a = []
    for j in range(0,20):
        soup2 = soup.find_all('td', class_="title")[j]
        soup3 = soup2.find('a')['href']
        a.append(soup3)
        
    for h in a:
            url = "http://kin.naver.com" + h
            res = req.urlopen(url)
            soup = BeautifulSoup(res,"html.parser")
            
            l = soup.select("#contents_layer_0 > div.end_content._endContents")
            
            for k in l:
                #print(len(k.get_text(strip = True)))
                shopping_text.append(k.get_text(strip = True))
    print(len(shopping_text))



################################################################################
shopping_text = set(shopping_text)
shopping_text = list(shopping_text)
shopping_text

with open('c:/python/shopping_text.csv','w',encoding = 'utf-8') as f:
    writer = csv.writer(f, delimiter=',')
    for y in shopping_text:
        writer.writerow([y])
      
df = pd.read_csv("c:/python/shopping_text.csv", names=['text'])



shopping_title = set(shopping_title)
shopping_title = list(shopping_title)
shopping_title

with open('c:/python/shopping_title.csv','w',encoding = 'utf-8') as f:
    writer = csv.writer(f, delimiter=',')
    for y in shopping_title:
        writer.writerow([y])
      
df = pd.read_csv("c:/python/shopping_title.csv", names=['title'])
################################################################################


"""




