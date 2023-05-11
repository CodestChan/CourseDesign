# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取原始图像
# img = cv2.imread(r'D:\Pycharm\pycharm\bathroom.jpg', cv2.IMREAD_UNCHANGED)
# img=cv2.resize(img,(640,400))
# print(img.shape)
#
# # 图像二维像素转换为一维
# data = img.reshape((-1, 3))
# data = np.float32(data)
#
# # 定义迭代条件 (type,max_iter,epsilon)
# criteria = (cv2.TERM_CRITERIA_EPS +
#             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#
# # 设置标签
# flags = cv2.KMEANS_RANDOM_CENTERS
#
# # K-Means聚类 聚集成2类
# compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
#
# # 图像转换回uint8二维类型，并且用聚类中心值替代与其同簇内所有像素点的值
# centers2 = np.uint8(centers2)
# res1 = centers2[labels2.flatten()]
# dst2 = res1.reshape(img.shape)
#
# result= np.concatenate([img,dst2], axis = 1)
# cv2.imshow('demo', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2

img=cv2.imread(r'D:\Pycharm\pycharm\bathroom.jpg')
img=cv2.resize(img,(640,400))
row = img.shape[0]
col = img.shape[1]

def knn(data, iter, k):
    data = data.reshape(-1, 3)
    data = np.column_stack((data, np.ones(row * col)))
    # 1.随机产生初始簇心
    cluster_center = data[np.random.choice(row * col, k)]
    # 2.分类
    distance = [[] for i in range(k)]
    for i in range(iter):
        # 2.1距离计算
        for j in range(k):
            distance[j] = np.sqrt(np.sum((data - cluster_center[j]) ** 2, axis=1))
        # 2.2归类
        data[:, 3] = np.argmin(distance, axis=0)
        # 3.计算新簇心
        for j in range(k):
            cluster_center[j] = np.mean(data[data[:, 3] == j], axis=0)
    return data[:, 3]


if __name__ == "__main__":
    image_show = knn(img, 100, 1)
    image_show = image_show.reshape(row, col)
    cv2.imshow('result',image_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # image1=knn(img,100,1)
    # image1=image1.reshape(row, col)
    # image2=knn(img,100,2)
    # image2=image2.reshape(row, col)
    # image3=knn(img,100,3)
    # image3=image3.reshape(row, col)
    # image4=knn(img,100,4)
    # image4=image4.reshape(row, col)
    # pic1=[image1,image2]
    # pic2=[image3,image4]
    # result1=cv2.hconcat(pic1)
    # result2=cv2.hconcat(pic2)
    # result=cv2.vconcat([result1,result2])

