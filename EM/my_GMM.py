import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# 生成三个高斯模型的数据
def generate_data(num_data = 1000):
    data=[]

    mean1 = [2, 1]  # 均值
    cov1 = [[5, 4], [4, 5]]  # 协方差矩阵
    mean2 = [10, 10]
    cov2 = [[10, 1], [1, 2]]
    mean3 = [7, 0]
    cov3 = [[3, 0], [0, 4]]

    alpha = [0.5,0.3,0.2]  # 三个高斯分布的权重

    for i in range(num_data):
        if np.random.random(1) < alpha[0]:
            data.append(np.random.multivariate_normal(mean1, cov1, 1))
        elif alpha[0] <= np.random.random(1) < alpha[0]+alpha[1]:
            data.append(np.random.multivariate_normal(mean2, cov2, 1))
        else:
            data.append(np.random.multivariate_normal(mean3, cov3, 1))

    return np.asarray(data)


class GMM:
    def __init__(self, data, category):
        self.data = data   # 训练数据
        self.category = category    # 高斯模型个数
        self.Z = np.zeros([data.shape[0],category])  # 隐变量
        self.alpha = np.array([1/3,1/3,1/3])  # 高斯模型的权重
        self.mean = np.random.random([category,2])*10  # 高斯模型的均值参数
        self.sigma = np.array([[[1., 0.], [0., 1.]],[[1., 0.], [0., 1.]],[[1., 0.], [0., 1.]]])  # 高斯模型的协方差矩阵参数

    def e_step(self):
        for i in range(self.data.shape[0]):  # 第i个样本点
            sum = 0
            temp = np.zeros([self.category])

            for k in range(self.category):  # 第k个模型
                temp[k] =  self.alpha[k] * math.exp(
                    -0.5*np.dot(np.dot(np.squeeze((self.data[i]-self.mean[k])), np.linalg.pinv(self.sigma[k]) ),
                                np.transpose(self.data[i]-self.mean[k])))\
                    /(2 * math.pi) / (np.sqrt(np.linalg.det(self.sigma[k]))+0.01)

            sum = np.sum(temp)
            for k in range(self.category):
                self.Z[i, k] = temp[k] / sum  # 第i个样本属于第k个模型的概率

    def m_step(self):
        # alpha更新
        self.alpha = np.sum(self.Z, axis=0)/self.data.shape[0]

        # 均值、协方差更新
        for k in range(self.category):
            temp_mean = 0
            temp_sigma = 0
            for i in range(self.data.shape[0]):
                temp_mean += self.Z[i, k] * self.data[i]
                temp_sigma += self.Z[i, k] * np.dot(np.transpose(self.data[i]-self.mean[k]),(self.data[i]-self.mean[k]))

            temp_mean = temp_mean/self.alpha[k]/self.data.shape[0]
            temp_sigma = temp_sigma/self.alpha[k]/self.data.shape[0]
            self.mean[k] = np.squeeze(temp_mean)
            self.sigma[k] = temp_sigma

        print(self.alpha)
        print(self.mean)
        print(self.sigma)



def show_scatter(data, Z, iteration):
    data = np.squeeze(data)
    plt.subplot(4, 5, iteration)
    for i in range(data.shape[0]):
        max_index = np.argmax(Z[i])

        if max_index == 0:
            plt.scatter(data[i,0], data[i,1], color="red", linewidths=0.1)
        elif max_index == 2:
            plt.scatter(data[i,0], data[i,1], color="green", linewidths=0.1)
        else:
            plt.scatter(data[i, 0], data[i, 1], color="blue", linewidths=0.1)


def show_3D(gmm):
    plt.subplot(221)
    plt.title('random generated data')
    x,y = gmm.data.T
    plt.scatter(x, y, color="black", linewidths=0.1)


    data = np.squeeze(gmm.data)
    probility = np.zeros([data.shape[0]])  # 每个样本的概率密度
    max_index = np.zeros([data.shape[0]])  # 样本类别的序号

    plt.subplot(222)
    plt.title('classified data through EM')
    for i in range(data.shape[0]):
        max_index[i] = np.argmax(gmm.Z[i])

        if max_index[i] == 0:
            plt.scatter(data[i,0], data[i,1], color="red", linewidths=0.1)
        elif max_index[i] == 1:
            plt.scatter(data[i,0], data[i,1], color="green", linewidths=0.1)
        else:
            plt.scatter(data[i, 0], data[i, 1], color="blue", linewidths=0.1)

        for k in range(3):
            probility[i] += gmm.alpha[k] * math.exp(
                    -0.5*np.dot(np.dot(np.squeeze((gmm.data[i]-gmm.mean[k])), np.linalg.pinv(gmm.sigma[k]) ),
                                np.transpose(gmm.data[i]-gmm.mean[k])))\
                    /(2 * math.pi) / (np.sqrt(np.linalg.det(gmm.sigma[k])))

    color = ['red', 'green', 'blue']
    ax = plt.subplot(223, projection='3d')
    plt.title('3d view')
    for i in range(data.shape[0]):
        ax.scatter(data[i, 0], data[i, 1], probility[i], c=color[int(max_index[i])])
    plt.show()


if __name__ == "__main__":
    data  = generate_data()
    gmm = GMM(data, 3)

    for i in range(50):
        gmm.e_step()
        gmm.m_step()
        # show_scatter(data, gmm.Z, i+1)

    show_3D(gmm)






