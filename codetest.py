import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread(r'D:\Pycharm\pycharm\indoor.jpg', cv2.IMREAD_UNCHANGED)
img=cv2.resize(img,(800,510))
print(img.shape)

# 图像二维像素转换为一维
data = img.reshape((-1, 3))
data = np.float32(data)

# 定义迭代条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成2类
compactness, labels2, centers2 = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 图像转换回uint8二维类型，并且用聚类中心值替代与其同簇内所有像素点的值
centers2 = np.uint8(centers2)
res1 = centers2[labels2.flatten()]
dst2 = res1.reshape(img.shape)

result= np.concatenate([img,dst2], axis = 1)
cv2.imshow('demo', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
