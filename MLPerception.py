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
import copy

def init_param(sigma,size):
    W=np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            W[i][j]=rd.uniform(-sigma,sigma)
    return W

def soft_max(x):
    exp_x=np.exp(x)
    return exp_x/np.sum(exp_x)

def Relu(x):
    t=copy.deepcopy(x)
    t[t<0]=0
    return t

sigma=np.sqrt(6/104)
label_i=['B','M','E','S']
TRAIN=r'D:\Machinelearn\envs\pynlp\pyhanlp-master\pyhanlp\static\data\test\icwb2-data\training\pku_training.txt'

train_label=[]
with open(TRAIN,'r') as f:
    t_text=f.read()
    t_text=t_text.replace('\n',' ')
    test_split=t_text.split()
    for word in test_split:
        if len(word)==1 :
            train_label.append('S')
        else:
            train_label.append('B')
            for i in range(len(word[1:-1])):
                train_label.append('M')
            train_label.append('E')

train_w2v=[]
vec_set={}
test=t_text.replace(' ','')
with open('out_w2v.txt','r',encoding='utf-8') as f1:
    for vector in f1.readlines()[1:]:
        vector=vector.strip('\n')
        w2v=list(map(eval,vector[2:].split()))
        w2v=np.array(w2v)
        vec_set[vector[0]]=w2v

    for word in test:
        train_w2v.append(vec_set[word])

    train_w2v=np.array(train_w2v)


# train_w2v=[]
# with open('out_w2v.txt','r',encoding='utf-8') as f1:
#     for vector in f1.readlines()[1:]:
#         vector=vector.strip('\n')
#         temp=list(map(eval,vector[2:].split()))
#         train_w2v.append(temp)
# train_w2v=np.array(train_w2v)

# train_label=[]
# with open('train_labels.txt','r',encoding='utf-8') as f2:
#     for label in f2:
#         train_label.append(label[-2])

train_p= {}
with open('train_p.txt','r',encoding='utf-8') as f3:
    for p in f3.readlines():
        p=p.strip('\n')
        temp=list(map(eval,p[2:].split()))
        train_p[p[0]]=np.array(temp)

W1,b1=init_param(sigma,(12,100)),init_param(sigma,(12,1))
W2,b2=init_param(sigma,(12,12)),init_param(sigma,(12,1))
W3,b3=init_param(sigma,(4,12)),init_param(sigma,(4,1))
# for gen in range(10):
for i in range(train_w2v.shape[0]):
    z1=np.dot(W1,train_w2v[i].reshape(-1,1))+b1
    y1=Relu(z1)

    z2=np.dot(W2,y1)+b2
    y2=Relu(z2)

    z3=np.dot(W3,y2)+b3
    y3=soft_max(z3)

    temp_y=y3.ravel()
    if label_i[np.argmax(temp_y)] == train_label[i]:
        continue

    z1[z1>0]=1
    z1[z1<0]=0
    z2[z2>0]=1
    z2[z2<0]=0
    sig3=y3-train_p[test[i]].reshape(-1,1)
    sig2=np.dot(np.transpose(W3),sig3)*z2
    sig1=np.dot(np.transpose(W2),sig2)*z1
    # sig2=np.dot(np.transpose(W3),sig3)*np.gradient(np.tanh(z2),axis=0)
    # sig1=np.dot(np.transpose(W2),sig2)*np.gradient(np.tanh(z1),axis=0)

    W3-=0.2*np.dot(sig3,y2.reshape(1,-1))
    b3-=0.2*sig3
    W2-=0.2*np.dot(sig2,y1.reshape(1,-1))
    b2-=0.2*sig2
    W1-=0.2*np.dot(sig1,train_w2v[i].reshape(1,-1))
    b1-=0.2*sig1

data={'W1':W1,'b1':b1,'W2':W2,'b2':b2,'W3':W3,'b3':b3}
np.savez('train_para.npy',**data)







