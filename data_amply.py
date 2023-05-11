import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np

IMG=r'D:\Notepad\大三课程\深度学习\实验1资料\data_original\img'
LAB=r'D:\Notepad\大三课程\深度学习\实验1资料\data_original\lab'
NEWIMG=r'D:\Notepad\大三课程\深度学习\实验1资料\knife\JPEGImages'
NEWLAB=r'D:\Notepad\大三课程\深度学习\实验1资料\knife\Annotations'
change=lambda x:eval(x.text)

for i in range(232):
    img=plt.imread(IMG+'\9000'+'{:0>3}.jpg'.format(i+1))
    with open(LAB+'\9000'+'{:0>3}.xml'.format(i+1),'r') as f:
        data=f.read()
    soup=BeautifulSoup(data,features='xml')
    x_min = soup.findAll('xmin')
    y_min = soup.findAll('ymin')
    x_max = soup.findAll('xmax')
    y_max = soup.findAll('ymax')

    width = eval(soup.find('width').text)
    height=eval(soup.find('height').text)
    w=[width,height,width]

    imgt=img
    xmi=list(map(change,x_min))
    ymi=list(map(change,y_min))
    xma=list(map(change,x_max))
    yma=list(map(change,y_max))
    for j in range(3):
        img_n=np.rot90(imgt)
        plt.imsave(NEWIMG+'\9000'+'{:0>3}_{}.jpg'.format(i+1,j+1),img_n)
        for k in range(len(x_min)):
            xi=xmi[k]
            xa=xma[k]
            d1=abs(xi-w[j]/2)
            d2=abs(xa-w[j]/2)

            if xi<(w[j]/2) :
                xmin=xi+2*d1
            else:xmin=xi-2*d1
            if xa<(w[j]/2):
                xmax=xa+2*d2
            else:xmax=xa-2*d2

            x_min[k].string=str(ymi[k])
            x_max[k].string=str(yma[k])
            y_min[k].string=str(xmax)
            y_max[k].string=str(xmin)

            xmi[k] = ymi[k]
            xma[k] = yma[k]
            ymi[k] = xmax
            yma[k] = xmin

        wt=soup.find('width')
        ht=soup.find('height')
        wt.string,ht.string=ht.string,wt.string

        data_n=str(soup)
        with open(NEWLAB+'\9000'+'{:0>3}_{}.xml'.format(i+1,j+1),'w',encoding='utf-8') as f1:
            f1.write(data_n.replace('<?xml version="1.0" encoding="utf-8"?>\n',''))
        imgt=img_n

# plt.subplot(1,3,1)
# with open(NEWLAB+r'\9000006_1.xml','r') as f:
#     data=f.read()
# soup=BeautifulSoup(data,features='xml')
# im=plt.imread(NEWIMG+r'\9000006_1.jpg')
# xmin=list(map(change,soup.findAll('xmin')))
# ymin=list(map(change,soup.findAll('ymin')))
# xmax=list(map(change,soup.findAll('xmax')))
# ymax=list(map(change,soup.findAll('ymax')))
# for i in range(len(xmin)):
#     plt.scatter([xmin[i],xmax[i]],[ymin[i],ymax[i]],c='r',marker='o')
# plt.imshow(im)
#
# plt.subplot(1,3,2)
# with open(NEWLAB+r'\9000006_2.xml','r') as f:
#     data=f.read()
# soup=BeautifulSoup(data,features='xml')
# im=plt.imread(NEWIMG+r'\9000006_2.jpg')
# xmin=list(map(change,soup.findAll('xmin')))
# ymin=list(map(change,soup.findAll('ymin')))
# xmax=list(map(change,soup.findAll('xmax')))
# ymax=list(map(change,soup.findAll('ymax')))
# for i in range(len(xmin)):
#     plt.scatter([xmin[i],xmax[i]],[ymin[i],ymax[i]],c='r',marker='o')
# plt.imshow(im)
#
# plt.subplot(1,3,3)
# with open(NEWLAB+r'\9000006_3.xml','r') as f:
#     data=f.read()
# soup=BeautifulSoup(data,features='xml')
# im=plt.imread(NEWIMG+r'\9000006_3.jpg')
# xmin=list(map(change,soup.findAll('xmin')))
# ymin=list(map(change,soup.findAll('ymin')))
# xmax=list(map(change,soup.findAll('xmax')))
# ymax=list(map(change,soup.findAll('ymax')))
# for i in range(len(xmin)):
#     plt.scatter([xmin[i],xmax[i]],[ymin[i],ymax[i]],c='r',marker='o')
# plt.imshow(im)
#
# plt.show()
# plt.subplot(1,3,1)
# with open(NEWLAB+r'\9000001_1.xml','r') as f:
#     data=f.read()
# soup=BeautifulSoup(data,features='xml')
# im=plt.imread(NEWIMG+r'\9000001_1.jpg')
# xmin=eval(soup.find('xmin').text)
# ymin=eval(soup.find('ymin').text)
# xmax=eval(soup.find('xmax').text)
# ymax=eval(soup.find('ymax').text)
# plt.scatter([xmin,xmax],[ymin,ymax],c='r',marker='o')
# plt.imshow(im)
#
# plt.subplot(1,3,2)
# with open(NEWLAB+r'\9000001_2.xml','r') as f:
#     data=f.read()
# soup=BeautifulSoup(data,features='xml')
# im=plt.imread(NEWIMG+r'\9000001_2.jpg')
# xmin=eval(soup.find('xmin').text)
# ymin=eval(soup.find('ymin').text)
# xmax=eval(soup.find('xmax').text)
# ymax=eval(soup.find('ymax').text)
# plt.scatter([xmin,xmax],[ymin,ymax],c='r',marker='o')
# plt.imshow(im)
#
# plt.subplot(1,3,3)
# with open(NEWLAB+r'\9000001_3.xml','r') as f:
#     data=f.read()
# soup=BeautifulSoup(data,features='xml')
# im=plt.imread(NEWIMG+r'\9000001_3.jpg')
# xmin=eval(soup.find('xmin').text)
# ymin=eval(soup.find('ymin').text)
# xmax=eval(soup.find('xmax').text)
# ymax=eval(soup.find('ymax').text)
# plt.scatter([xmin,xmax],[ymin,ymax],c='r',marker='o')
# plt.imshow(im)
#
# plt.show()



