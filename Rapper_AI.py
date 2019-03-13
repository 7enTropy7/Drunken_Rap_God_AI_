# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:02:51 2019

@author: Unnikrishnan Menon
"""
import pandas as pd
import urllib.request as urllib2
from bs4 import BeautifulSoup
import re
from unidecode import unidecode
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam

site='http://metrolyrics.com/{}-lyrics-eminem.html'
file=pd.read_csv('eminem.csv')
names=pd.read_csv('Eminem_Rap_Names.csv')
name_list=names.iloc[:,0].values
print(name_list)
counter=1
for f in range(len(name_list)):
    lyrics=''
    page=urllib2.urlopen(site.format(name_list[f]))    
    soup=BeautifulSoup(page,'html.parser')
    lines=soup.find_all('p', attrs={'class': 'verse'})
    for i in lines:
        text=i.text.strip()
        text=re.sub(r"\[.*\]\n","",unidecode(text))
        if lyrics=='':
            lyrics=lyrics+text.replace('\n','|-|')
        else:
            lyrics=lyrics+'|-|'+text.replace('\n','|-|')
    file.at[f+1,'lyrics']=lyrics
    print(name_list[f])
    counter+=1
print('All Raps Saved Successfully!')
file.to_csv('eminem.csv',sep=',',encoding='utf-8')
file=pd.read_csv('eminem.csv')
for i,row in file["lyrics"].iteritems():
    text=text+str(row).lower()
next_chars=[]
sentences=[]
u_chars=sorted(list(set(text)))
c_i=dict((c,i) for i,c in enumerate(u_chars))
i_c=dict((i,c) for i,c in enumerate(u_chars))
for i in range(0,len(text)-40,1):
    sentences.append(text[i:i+40])
    next_chars.append(text[i+40])
x=np.zeros((len(sentences),40,len(u_chars)), dtype=np.bool)
y=np.zeros((len(sentences), len(u_chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, c_i[char]] = 1
    y[i, c_i[next_chars[i]]] = 1
model = Sequential()
model.add(LSTM(128, input_shape=(40,len(u_chars))))
model.add(Dense(len(u_chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=128, epochs=30)
model.save('my_model.h5')
model=load_model('my_model.h5')
def converter(p,t=1.0):
    p=np.asarray(p).astype('float64')
    p=np.log(p) / t
    exp_p=np.exp(p)
    p=exp_p/np.sum(exp_p)
    probs = np.random.multinomial(1,p,1)
    return np.argmax(probs)
def rap_god():
    rap=''
    ip=input("Gimme some shit to start with : ")
    sentence=('{0:0>' + str(40) + '}').format(ip).lower()
    rap+=ip 
    print("\nNow listen to mah Rap: \n") 
    print(ip,end='')
    for i in range(500):
        x_pred=np.zeros((1,40,len(u_chars)))
        for t,u in enumerate(sentence):
            if u!='0':
                x_pred[0,t,c_i[u]]=1.
        p=model.predict(x_pred, verbose=0)[0]
        next_index=converter(p,t=0.2)
        next_char=i_c[next_index]
        rap+=next_char
        sentence=sentence[1:]+next_char
        print(next_char,end='')
        if next_char=='\n':
            continue
rap_god()