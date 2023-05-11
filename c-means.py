import matplotlib.pyplot as plt
import numpy as np

def distance(cor,center):
    ou=[]
    for i in range(4):
        t=np.linalg.norm(cor-center[i])
        ou.append(t)
    return ou.index(min(ou))

data=[]
f=open(r'D:\Notepad\大三课程\模式识别\data.txt','r',encoding='utf-8')
for cor in f.readlines():
    cor=cor.strip('\n')
    temp=list(map(eval,cor.split()))
    data.append(temp)
data=np.array(data)
f.close()

center=[[-3,1],[2,4],[-1,-0.5],[2,-3]]
center=np.array(center)

cluster={0:[],1:[],2:[],3:[]}
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.scatter(center[:, 0], center[:, 1], c='r', marker='o')
    plt.scatter(data[:, 0], data[:, 1], c='b', marker='o')
    for cor in data:
        ind=distance(cor,center)
        cluster[ind].append(cor)
    for j in range(4):
        temp=np.array(cluster[j])
        if len(cluster[j])!=0:
            center[j]=np.mean(temp,axis=0)
        cluster[j].clear()

plt.show()

