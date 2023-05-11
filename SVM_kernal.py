# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
#
# def kernel(x1, x2, kertype='linear'):
#     if kertype == 'linear':
#         return np.dot(x1.T, x2)
#     elif kertype == 'gaussian':
#         sigma = 0.1
#         return np.exp(-np.linalg.norm(x1-x2)**2/(2*sigma**2))
#
# def svmTrain(X, Y, kertype='linear', C=10):
#     n = Y.shape[1]
#     H = np.dot(Y.T, Y) * kernel(X, X, kertype)
#     f = -np.ones(n)
#     F=f.reshape(60,1)
#     A = np.empty((0, n))
#     b = np.empty((0, 1))
#     Aeq = Y
#     beq = np.zeros((1, 1))
#     lb = np.zeros((n, 1))
#     ub = C*np.ones((n, 1))
#     a0 = np.zeros((n, 1))
#     options = {'disp': False}
#     res = minimize(lambda a: 0.5*np.dot(a.T, np.dot(H, a)) - np.dot(a.T, F), a0, method='SLSQP',
#                    bounds=[(0, C)]*n, constraints={'type': 'eq', 'fun': lambda a: np.dot(a, Y)}, options=options)
#     sv = res.x > 1e-5
#     a = res.x[sv]
#     sv_x = X[:, sv]
#     sv_y = Y[:, sv]
#     b = np.mean(sv_y - np.dot(a*sv_y.T, kernel(sv_x, sv_x, kertype)))
#     return {'kernel': kertype, 'Xsv': sv_x, 'Ysv': sv_y, 'alpha': a, 'b': b}
#
# def svmTest(svm, X, Y, kertype='linear'):
#     nt = X.shape[1]
#     Ypred = np.zeros((1, nt))
#     for i in range(nt):
#         Ypred[0, i] = np.sum(svm['alpha']*svm['Ysv']*kernel(svm['Xsv'], X[:, i], kertype)) + svm['b']
#     return {'kernel': kertype, 'Y': Ypred}
#
# np.random.seed(6)
# n = 30
# x1 = np.random.randn(2, n)
# y1 = np.ones((1, n))
# x2 = 4 + np.random.randn(2, n)
# y2 = -np.ones((1, n))
# fig, ax = plt.subplots()
# ax.plot(x1[0,:], x1[1,:], 'bs', x2[0,:], x2[1,:], 'k+')
# ax.axis([-3, 8, -3, 8])
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# X = np.hstack((x1, x2))
# Y = np.hstack((y1, y2))
# svm = svmTrain(X, Y, 'linear', 10)
# ax.plot(svm['Xsv'][0,:], svm['Xsv'][1,:], 'ro')
# xx1, xx2 = np.meshgrid(np.arange(-2, 7, 0.05), np.arange(-2, 7, 0.05))
# Xt = np.vstack((np.reshape(xx1, (1, -1)), np.reshape(xx2, (1, -1))))
# Yt = np.ones((1, Xt.shape[1]))
# result = svmTest(svm, Xt, Yt, 'linear')

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

np.random.seed(6)
n = 30
x1 = np.random.randn(2, n)
y1 = np.ones((1, n))
x2 = 4 + np.random.randn(2, n)
y2 = -np.ones((1, n))

X_train=np.concatenate((x1.T,x2.T),axis=0)
y_train=np.concatenate((y1,y2),axis=1).ravel()

svm1=SVC(kernel='linear')
svm1.fit(X_train,y_train)
svm2=SVC(kernel='rbf')
svm2.fit(X_train,y_train)

plt.subplot(1,2,1)
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svm1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
for vector in svm1.support_vectors_:
    plt.scatter(vector[0],vector[1],s=100,edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM Classification(linear)')

plt.subplot(1,2,2)
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svm2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
for vector in svm2.support_vectors_:
    plt.scatter(vector[0],vector[1],s=100,edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM Classification(rbf)')
plt.show()

# w=svm.coef_
# b=svm.intercept_
# svm.support_
# svm.support_vectors_
#
# x=np.linspace(-2,6,100)
# plt.scatter(x1[0],x1[1],c='r',marker='o')
# plt.scatter(x2[0],x2[1],c='b',marker='o')
# plt.plot(x,)
# plt.show()

