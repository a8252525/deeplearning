# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:26:39 2019

@author: ideapad 320
"""
import numpy as np
def relu(x):
    return np.maximum(x, 0)
def Drelu(x):
    return (x>0).astype('float32')
def softmax(x):
    t=np.exp(x)
    return t/t.sum()
L_history = []
L_rate= 0.03
A = np.random.normal(size=(10,4))
b = np.random.normal(size=(10,1))
C = np.random.normal(size=(3,10))
d = np.random.normal(size=(3,1))
for t in range(20000):
    i=np.random.randint(0,16)
    x = np.array([[i%2], [(i>>1)%2], [(i>>2)%2], [(i>>3)%2]])
    y=i%3
    U=(A@x)+b
    #(10,1)=(10,4)@(4,1)+(10,1)
    q=softmax(C@U+d)
    #(3,1)=(3,10)@(10,1)+(3,1)
    L=-np.log(q[y])
    L_history.append(L)
    p = np.eye(3)[y][:, None]
    grad_d=q-p
    #(3,1)
    grad_C = grad_d @ U.T
    #(3,10)=(3,1)@(1,10)
#    grad_b = (C.T @ grad_d ) * Drelu(A@x+b)
    grad_b = (C.T @ grad_d ) * Drelu(A@x+b)
    #(10,1)=(10,3)@(3,1)*(10,1)
    #Don't forget the diffenence of @ and *
    grad_A = grad_b @ x.T
    #(10,4)=(10,1)@(1,4)
    A -= L_rate * grad_A
    b -= L_rate* grad_b
    C -= L_rate * grad_C
    d -= L_rate * grad_d