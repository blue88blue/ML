import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random



class logistic_regression:
    def __init__(self, num_features):
        self.w = np.random.randn(num_features, 1)
        self.b = np.random.randn(1, 1)
        self.wb = np.row_stack([self.w, self.b])

    def predict(self, x, stack=True):
        if stack:
            x = np.column_stack([x, np.ones(x.shape[0])])
        y_pred = 1 / (np.exp(-np.dot(x, self.wb)) + 1)
        return y_pred

    def fit(self, train_x, train_y, samples_weight=None, weight=False):
        self.wb = np.random.random(self.wb.shape)
        train_x = np.column_stack([train_x, np.ones(train_x.shape[0])])

        for i in range(10000):
            if weight == False:
                d_wb = np.dot(np.transpose(train_x), (self.predict(train_x, stack=False) - train_y)) / train_x.shape[0]
            else:
                d_wb = np.dot(np.transpose(train_x),
                              (self.predict(train_x, stack=False) - train_y) * samples_weight / np.sum(samples_weight))
            self.wb = self.wb - 0.01 * d_wb



def load_data(data_dir='./data.csv'):
    dataset = pd.read_csv(data_dir)
    dataset = dataset.sample(frac=1)
    data_x = dataset.iloc[:,1:-1].values
    data_y = dataset.iloc[:,-1].values.reshape([-1,1])

    # for i in range(data_x.shape[0]):
    #     if data_y[i][0] == 0:
    #         plt.scatter(data_x[i][0],data_x[i][1], color = 'red')
    #     else:
    #         plt.scatter(data_x[i][0],data_x[i][1], color = 'green')
    return data_x, data_y


def creat_data(num, width=100, hight=100):
    data_x = []
    data_y = []
    weight=[]
    for num in range(num):
        i = random.randint(0, 99)
        j = random.randint(0, 99)
        data_x.append([i, j])
        # if (i-hight/2-40)**2+(j-width/2-40)**2<5000:
        if (i - hight / 2 - 40) ** 2 + (j - width / 2 - 40) ** 2 < 6400 and (i - hight / 2 - 40) ** 2 + (
                j - width / 2 - 40) ** 2 > 900:
            data_y.append([1])
        else:
            data_y.append([0])
        if -i+100 < j:
            weight.append([10])
        else:
            weight.append([1])
    return np.asarray(data_x)/100, np.asarray(data_y), np.asarray(weight)




if __name__ == '__main__':

    # data_x, data_y = load_data()
    data_x, data_y ,weight= creat_data(500)


    model = logistic_regression(2)
    model.fit(data_x, data_y, samples_weight=weight, weight=True)
    pred = model.predict(data_x)


    # 构造数据
    x = [random.uniform(0, 1) for i in range(10000)]
    x = np.asarray(x).reshape([-1, 2])
    y_p = model.predict(x)

    # 显示预测结果
    for i in range(x.shape[0]):
        if y_p[i][0] < 0.5:
            plt.scatter(x[i][0], x[i][1], color='pink', linewidths=0.2)
        else:
            plt.scatter(x[i][0], x[i][1], color='lightgreen', linewidths=0.2)

    for i in range(data_x.shape[0]):  # 训练集数据
        if data_y[i][0] < 0.5:
            if weight[i][0]>1:
                plt.scatter(data_x[i][0], data_x[i][1], color='darkred', linewidths=0.2)
            else:
                plt.scatter(data_x[i][0], data_x[i][1], color='red', linewidths=0.2)
        else:
            if weight[i][0]>1:
                plt.scatter(data_x[i][0], data_x[i][1], color='darkgreen', linewidths=0.2)
            else:
                plt.scatter(data_x[i][0], data_x[i][1], color='green', linewidths=0.2)
    plt.show()