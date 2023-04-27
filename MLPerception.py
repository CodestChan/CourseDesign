# import json
#
# def sum_dict(dic:dict):
#     sum=0
#     for key in dic:
#         sum+=dic[key]
#     return sum
#
# f1=open('words_label.json','r')
# words_label=json.load(f1)
# f1.close()
#
# f2=open('train_labels.txt','r',encoding='utf-8')
# train_labels=f2.readlines()
# f2.close()
#
# condition=['B','M','E','S']
# hoods=[]
# f3=open('train_p.txt','w',encoding='utf-8')
# for label in train_labels:
#     hood=[]
#     sum_h=sum_dict(words_label[label[0]])
#     hood.append(label[0])
#     for i in condition:
#         p=words_label[label[0]].get(i,0)
#         hood.append(str(p/sum_h))
#     hoods.append(hood)
# for hoo in hoods:
#     f3.write(' '.join(hoo)+'\n')
# f3.close()

import numpy as np
import random as rd

def init_param(sigma,size):
    W=np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            W[i][j]=rd.uniform(-sigma,sigma)
    return W

def soft_max(x):
    exp_x=np.exp(x)
    return exp_x/np.sum(exp_x)

sigma=np.sqrt(6/104)
label=['B','M','E','S']

train_w2v=[]
with open('out_w2v.txt','r',encoding='utf-8') as f1:
    for vector in f1.readlines()[1:]:
        vector=vector.strip('\n')
        temp=list(map(eval,vector[2:].split()))
        train_w2v.append(temp)
train_w2v=np.array(train_w2v)

train_label=[]
with open('train_labels.txt','r',encoding='utf-8') as f2:
    for label in f2:
        train_label.append(label[-2])

train_p=[]
with open('train_p.txt','r',encoding='utf-8') as f3:
    for p in f3.readlines():
        p=p.strip('\n')
        temp=list(map(eval,p[2:].split()))
        train_p.append(temp)
train_p=np.array(train_p)

W1,b1=init_param(sigma,(12,100)),init_param(sigma,(12,1))
W2,b2=init_param(sigma,(12,12)),init_param(sigma,(12,1))
W3,b3=init_param(sigma,(4,12)),init_param(sigma,(12,1))

for i in range(train_w2v.shape[0]):
    z1=np.dot(W1,train_w2v[i].reshape(-1,1))+b1
    y1=np.tanh(z1)
    z2=np.dot(W2,y1)+b2
    y2=np.tanh(z2)
    z3=np.dot(W3,y2)+b3
    y3=soft_max(z3)

    temp_y=y3.ravel






