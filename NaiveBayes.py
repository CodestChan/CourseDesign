import numpy as np
import matplotlib.pyplot as plt

# def gaussion(x,mu,sigma):
#     return np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
#
# x = [-3.9847, -3.5549, -1.2401, -0.9780, -0.7932, -2.8531, -2.7605, -3.7287,
#      -3.5414, -2.2692, -3.4549, -3.0752, -3.9934, 2.8792, -0.9780, 0.7932,
#      1.1882, 3.0682, -1.5799, -1.4885, -0.7431, -0.4221, -1.1186, 4.2532]
#
# x_d=np.array(x)
# pw1,pw2=0.9,0.1
# R=np.array([[0,4],[2,0]])
# e1,a1=-2,0.5
# e2,a2=2,2
#
# pw1_x=np.zeros(len(x_d))
# pw2_x=np.zeros(len(x_d))
# for i in range(len(x_d)):
#     pw1_x[i]=gaussion(x_d[i],-2,0.5)
#     pw2_x[i]=gaussion(x_d[i],2,2)

# px_w1=np.zeros(len(x_d))
# px_w2=np.zeros(len(x_d))
# px_w1=pw1_x*pw1/np.sum(pw1_x*pw1+pw2_x*pw2)
# px_w2=pw2_x*pw2/np.sum(pw1_x*pw1+pw2_x*pw2)
# result_e=np.zeros(len(x_d))
# result_e[px_w1<px_w2]=1

# plt.subplot(1,2,1)
# x_h=np.arange(len(x_d))
# plt.plot(x_d,px_w1)
# plt.plot(x_d,px_w2)
# plt.title('min error')
# plt.scatter(x_h,result_e+1)

#min_risk
# Ra1_x=0*px_w1+4*px_w2
# Ra2_x=2*px_w1+0*px_w2
# result_r=np.zeros(len(x_d))
# result_r[Ra1_x>Ra2_x]=1

# plt.subplot(1,2,2)
# plt.plot(x_d,Ra1_x)
# plt.plot(x_d,Ra2_x)
# plt.title('min risk')
# # plt.scatter(x_h,result_e+1)
# plt.show()

a=np.array([1,2,3,4]).reshape(-1,1)
b=a.ravel()
print(a)
print(b)
print(b.index(3))
