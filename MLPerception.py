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



