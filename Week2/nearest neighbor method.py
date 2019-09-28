# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:40:34 2019

@author: ideapad 320
"""
import numpy as np
import os
import urllib
from urllib.request import urlretrieve 
print('正在下載')
dataset = 'mnist.pkl.gz'
if not os.path.isfile(dataset):
        origin = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
        print('Downloading data from %s' % origin)
        urlretrieve(origin, dataset)
# 下載完了
import gzip #用來處理壓縮檔
import pickle #
with gzip.open(dataset, 'rb') as f:#gzip.open來打開壓縮檔
    train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
#run -i q_see_mnist_data.py 這行是另外的.py，跑在console，可以看到裡面的[0]是圖片用浮點數形式，[1]是他的答案
# 確認完資料
train_X, train_y = train_set
test_X, test_y = test_set
train_y[:20]#就是剛剛的train_set[1]的部分

#完成下載，並放到train_X,test_X中
print('Running...')

train_X  = train_X / np.linalg.norm(train_X, axis=1, keepdims=True)
test_X  = test_X / np.linalg.norm(test_X, axis=1, keepdims=True)
A = test_X @ train_X.T
#做出cos similarity

predict=train_y[A.argmax(axis=1)]
count=0
for i,j in enumerate(predict):
    if j==test_y[i]:
        count+=1
print('準確率:{}'.format(count/len(predict)))