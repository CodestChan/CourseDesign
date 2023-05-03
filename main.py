from gensim.models import Word2Vec
import numpy as np
TRAIN=r'D:\Machinelearn\envs\pynlp\pyhanlp-master\pyhanlp\static\data\test\icwb2-data\training\pku_training.txt'
TEST=r'D:\Machinelearn\envs\pynlp\pyhanlp-master\pyhanlp\static\data\test\icwb2-data\testing\pku_test.txt'
test_vector='test_vector.txt'
out_w2v='out_w2v.txt'
f1=open(TEST,'r')
sentences=[]
for line in f1:
    line=line.strip('\n')
    sentence=''.join(line.split())
    word_list=[word for word in sentence]
    sentences.append(word_list)
model=Word2Vec(sentences,vector_size=100,window=5,min_count=1,workers=4)
# f=open('output_file.txt', 'w', encoding='utf-8')
# model.save_word2vec_format(f, binary=False)
# f.close()
# model.save(out_w2v)
# model = Word2Vec.load('out_w2v.txt')
model.wv.save_word2vec_format(test_vector, binary=False)
f1.close()

import json

TRAIN=r'D:\Machinelearn\envs\pynlp\pyhanlp-master\pyhanlp\static\data\test\icwb2-data\training\pku_training.txt'
TEST=r'D:\Machinelearn\envs\pynlp\pyhanlp-master\pyhanlp\static\data\test\icwb2-data\gold\pku_test_gold.txt'

f=open(TEST,'r')

words={}
text=f.read()
text=text.replace('\n',' ')
text_list=text.split()

for word in text_list:
    if len(word)==1:
        words[word]=words.get(word,dict())
        words[word]['S']=words[word].get('S',0)
        words[word]['S']+=1
    else:
        words[word[0]]=words.get(word[0],dict())
        words[word[0]]['B']=words[word[0]].get('B',0)
        words[word[0]]['B']+=1

        for i in word[1:-1]:
            words[i] = words.get(i, dict())
            words[i]['M'] = words[i].get('M', 0)
            words[i]['M'] += 1

        words[word[-1]]=words.get(word[-1],dict())
        words[word[-1]]['E']=words[word[-1]].get('E',0)
        words[word[-1]]['E']+=1

with open('test_label.json','w') as f:
    json.dump(words,f)

f.close()

import json

f1=open('test_label.json','r')
data=json.load(f1)
f1.close()

labels=[]
f2=open('test_vector.txt','r',encoding='utf-8')
outvectors=f2.readlines()
for vector in outvectors[1:]:
    if vector[0] in data:
        labels.append((vector[0],max(data[vector[0]])))
f2.close()

f3=open('test_gold.txt','w',encoding='utf-8')
for word in labels:
    f3.write(' '.join(word)+'\n')
f3.close()

import numpy as np
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

trans=np.zeros((4,4))
l=['B','M','E','S']
for i in range(1,len(train_label)):
    f=l.index(train_label[i-1])
    s=l.index(train_label[i])
    trans[f,s]+=1

trans/=len(train_label)-1
np.save('trans.npy',trans)

data=np.load('trans.npy')
print(data)

import matplotlib.pyplot as plt
x=np.linspace(-5,5,100)
plt.subplot(1,2,1)
y1=1/(1+np.exp(-x))
plt.plot(x,y1)
plt.subplot(1,2,2)
y2=np.tanh(x)
plt.plot(x,y2)
plt.show()