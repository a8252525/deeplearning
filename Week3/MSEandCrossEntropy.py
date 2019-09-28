# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:45:46 2019

@author: ideapad 320
"""
import numpy as np
import math
import gzip #用來處理壓縮檔
import pickle #

def relu(x):
    return np.maximum(x, 0)
def Drelu(x):
    return (x>0).astype('float32')
def softmax(x):
    t=np.exp(x)
    return t/t.sum()


    

dataset = 'mnist.pkl.gz'
with gzip.open(dataset, 'rb') as f:#gzip.open來打開壓縮檔
    train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
#run -i q_see_mnist_data.py 這行是另外的.py，跑在console，可以看到裡面的[0]是圖片用浮點數形式，[1]是他的答案
# 確認完資料
train_X, train_y = train_set
test_X, test_y = test_set
print('Training with loss function:cross entropy...')

test_X=test_X[...,None]

W1=np.random.normal(size=(50,784))
b1=np.random.normal(size=(50,1))
W2=np.random.normal(size=(10,50))
b2=np.random.normal(size=(10,1))
LR=0.008
L_history=[]
for i in range(20):
    idx=np.random.choice(5000,5000,replace=False)
    for j in idx:
        x=train_X[j]
        y=train_y[j]
        x=x.reshape(784,1)
        U=relu(W1@x+b1)
        V=softmax(W2@U+b2)
        L=-np.log(V[y])[0]
        L_history.append(L)
        p=np.eye(10)[y][:,None]
        grad_b2=V-p
        grad_W2=grad_b2@U.T
        grad_b1=(W2.T@grad_b2)*Drelu(W1@x+b1)
        grad_W1=grad_b1@x.T
        W1-=LR*grad_W1
        b1-=LR*grad_b1
        W2-=LR*grad_W2
        b2-=LR*grad_b2
    
    accuracy=((W2@relu(W1@test_X+b1)+b2).argmax(axis=1).ravel()==test_y).mean()
    print(accuracy)

    total_test=1000
    predict=[]
    for i in range(total_test):
        x=test_X[i].reshape(784,1)
        y=test_y[i]
        x=x.reshape(784,1)
        U=relu(W1@x+b1)
        V=(W2@U+b2)
        predict.append(V.argmax())
    accuracy=(predict==test_y[:total_test]).mean()
    print('Accuracy with Cross entropy: {}'.format(accuracy))
    print('bug')

W1=np.random.normal(size=(50,784))
b1=np.random.normal(size=(50,1))
W2=np.random.normal(size=(10,50))
b2=np.random.normal(size=(10,1))
LR=0.015
L_history=[]
for i in range(20):
    idx=np.random.choice(200,200,replace=False)
    for j in idx:
        x=train_X[j]
        y=train_y[j]
        x=x.reshape(784,1)
        U=relu(W1@x+b1)
        V=(W2@U+b2)
        p=np.eye(10)[y][:,None]
        L=math.sqrt(((V-p)*(V-p)).sum()/len(V))
        L_history.append(L)
        grad_loss=((((V-p)*(V-p)).sum()/len(V))**(-1/2))/(2*len(V)
        grad_b2= grad_loss
        grad_W2=grad_b2@U.T
        grad_b1=(W2.T@grad_b2)*Drelu(W1@x+b1)
        grad_W1=grad_b1@x.T
        W1-=LR*grad_W1
        b1-=LR*grad_b1
        W2-=LR*grad_W2
        b2-=LR*grad_b2
    
    accuracy=((W2@relu(W1@test_X+b1)+b2).argmax(axis=1).ravel()==test_y).mean()
    print(accuracy)



        
        