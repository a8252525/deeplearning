# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:32:53 2019

@author: ideapad 320
"""
import numpy as np
def softmax(x):
    t=np.exp(x)
    return t/t.sum()
def relu(x):
    return np.maximum(x, 0)

#mod4
mod4count=0
W=np.array([[-1,-1,0,0],[2,-1,-1,-1],[-1,2,-1,-1],[1,1,-1,-1]])
b=np.array([0,0,0,0])
for i in range(16):
    x=np.array([i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2])
    ans=W@x+b
    if ans.argmax()==i%4:
        mod4count+=1
print('Mod4 accuracy:{}'.format(mod4count/16))

#-------------------------------------------

mod3count=0
W=np.array([[0,0,0,0], [1,-1,1,-1],[-1,1,-1,1],[0,3,0,3],[3,0,0,3],])
b=np.array([0.5,0,0,-5,-5])
W2=np.array([[1,0,0,0,0], 
           [0,1,0,1,0], 
           [0,0,1,0,1],
          ])
b2=np.array([0,0,0])
for i in range(16):
    x=np.array([i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2])
    ans=softmax(W2@(relu(W@x+b))+b2)
    if ans.argmax()==i%3:
        mod3count+=1
print('Mod3 accuracy:{}'.format(mod3count/16))


#---------------------
count=0
A=np.array([[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1],[1,0,0,1,0,0,1,0,0],
              [0,1,0,0,1,0,0,1,0],[0,0,1,0,0,1,0,0,1],[1,0,0,0,1,0,0,0,1],[0,0,1,0,1,0,1,0,0]])
b=np.array([-2,-2,-2,-2,-2,-2,-2,-2])
C=np.array([[-1,-1,-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1,1,1]])
d=np.array([0, 0])
for i in range(10):
    table = np.random.randint(0,2, size=(3,3))

#    for i in table:
#
#        temp=''
#        for j in i:
#            if j==1:
#                temp+='X'
#            else:
#                temp+='_'
#        print('{}\n'.format(temp))
#   老師的寫法: print( "\n".join("".join("_X"[k] for k in  board[j]) for j in range(3))) 簡潔好看

    if (table.all(axis=0)).any() or (table.all(axis=1)).any() or table[::3].all() or table[2::2].all():
        ans=1
    else:
        ans=0
    x=np.array(table.flatten())
    q=softmax(C@relu(A@x+b)+d)
    if q.argmax()==ans:
        count+=1
    #print("q={}\n".format(q.argmax()))
print('Tic Tac Toe accuracy:{}'.format(count/10))

