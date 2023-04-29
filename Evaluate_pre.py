import numpy as np
import copy

data=np.load('train_para.npy.npz')
label_p=['B','M','E','S']
TEST=r'D:\Machinelearn\envs\pynlp\pyhanlp-master\pyhanlp\static\data\test\icwb2-data\testing\pku_test.txt'
GOLD=r'D:\Machinelearn\envs\pynlp\pyhanlp-master\pyhanlp\static\data\test\icwb2-data\gold\pku_test_gold.txt'

def soft_max(x):
    exp_x=np.exp(x)
    return exp_x/np.sum(exp_x)

def Relu(x):
    t=copy.deepcopy(x)
    t[t<0]=0
    return t

with open(TEST,'r') as f1:
    test=f1.read()
    test=test.replace('\n','')

vec_set= {}
test_w2v=[]
with open('test_vector.txt','r',encoding='utf-8') as f2:
    for vector in f2.readlines()[1:]:
        vector=vector.strip('\n')
        w2v=list(map(eval,vector[2:].split()))
        w2v=np.array(w2v)
        vec_set[vector[0]]=w2v

    for word in test:
        test_w2v.append(vec_set[word])

    test_w2v=np.array(test_w2v)

gold_label=[]
with open(GOLD,'r') as f3:
    gold=f3.read()
    gold=gold.replace('\n',' ')
    gold_split=gold.split()
    for word in gold_split:
        if len(word)==1 :
            gold_label.append('S')
        else:
            gold_label.append('B')
            for i in range(len(word[1:-1])):
                gold_label.append('M')
            gold_label.append('E')

W1,b1=data['W1'],data['b1']
W2,b2=data['W2'],data['b2']
W3,b3=data['W3'],data['b3']
pre_label=[]
for i in range(test_w2v.shape[0]):
    z1=np.dot(W1,test_w2v[i].reshape(-1,1))+b1
    y1=np.tanh(z1)
    y1[y1<0]=0

    z2=np.dot(W2,y1)+b2
    y2=np.tanh(z2)
    y2[y2<0]=0

    z3=np.dot(W3,y2)+b3
    y3=soft_max(z3)

    temp_y = y3.ravel()
    # pre_ind=np.argmax(temp_y)
    # pre_label.append(label_p[pre_ind])
    if i==0:
        pre_label.append('B')
    elif pre_label[i-1] in ['B','M']:
        temp_y[0]=temp_y[3]=-1
        pre_ind=np.argmax(temp_y)
        pre_label.append(label_p[pre_ind])
    elif pre_label[i-1] in ['E','S']:
        temp_y[1:3]=-1
        pre_ind=np.argmax(temp_y)
        pre_label.append(label_p[pre_ind])

print(pre_label)
