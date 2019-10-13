from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
from copy import deepcopy
import matplotlib.pyplot as plt
import random
from math import sqrt



def creat_data(num, width=100, hight=100):
    data_x = []
    data_y = []
    for num in range(num):
        i = random.randint(0, 99)
        j = random.randint(0, 99)
        data_x.append([i, j])
        # if (i-hight/2-40)**2+(j-width/2-40)**2<5000 :
        # if (i - hight / 2 - 40) ** 2 + (j - width / 2 - 40) ** 2 < 5000:
        if j > i-20 and j< -i+120:
            data_y.append([1])
        else:
            data_y.append([0])
    return np.asarray(data_x)/100, np.asarray(data_y)


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

    def fit(self, train_x, train_y, samples_weight=None, weight=True):
        self.wb = np.random.random(self.wb.shape)
        train_x = np.column_stack([train_x, np.ones(train_x.shape[0])])
        for i in range(50000):
            if weight == False:
                d_wb = np.dot(np.transpose(train_x), (self.predict(train_x, stack=False) - train_y)) / train_x.shape[0]
            else:
                d_wb = np.dot(np.transpose(train_x),
                              (self.predict(train_x, stack=False) - train_y) * samples_weight)
            self.wb = self.wb - 0.01 * d_wb



class adaboost:
    def __init__(self, max_classifiers, classifier, num_samples):
        self.max_classifiers = max_classifiers
        self.classifiers = []
        self.classifiers_weight = []
        self.classifier = classifier
        self.samples_weight = np.ones([num_samples, 1]) / num_samples  # 初始化权重为1

    def update_samples_weight(self, x, y):
        y_prob = self.classifier.predict(x)  # 测试当前分类器

        y_pred = np.where(y_prob > 0.5, 1, 0)
        misclassified_samples = np.where(y_pred == y, 0, 1)  # 分类错误的样本记为1
        error_rate = np.sum(self.samples_weight * misclassified_samples) / np.sum(self.samples_weight)

        d = np.sqrt((1-error_rate)/error_rate)
        self.classifiers_weight.append(np.log(d))  # 分类器的权重

        scale = np.where(misclassified_samples == 1, d, 1/d).reshape([-1,1])  # 若分类错误的样本权重×d,分类正确的样本权重/d
        self.samples_weight *= scale

        print('error_rate = %f    d = %f    weight=%f' % (error_rate, d, self.classifiers_weight[-1]), end='')


    def train(self,x_train, y_train, x_test, y_test):
        train_acc = []
        test_acc = []

        for i in range(self.max_classifiers):  # 循环生成多个弱分类器

            self.classifier.fit(x_train, y_train, samples_weight=self.samples_weight)  # 根据样本权重训练
            self.classifiers.append(deepcopy(self.classifier))  # 得到一个分类器。深度拷贝，拷贝当前对象及其所有子对象
            self.update_samples_weight(x_train, y_train)  # 计算训练错误率 更新样本权重

            # 计算当前所有分类器加在一起的分类准确率
            train_acc.append(self.cal_acc(x_train, y_train))
            test_acc.append(self.cal_acc(x_test, y_test))
            print("    train acc:%f    test acc:%f" %(train_acc[i], test_acc[i]))
            print(np.max(self.samples_weight), np.min(self.samples_weight))

        # 训练结束，显示acc曲线
        plt.xlabel('iterations')
        plt.ylabel('acc')
        plt.plot(np.arange(self.max_classifiers), train_acc, color='green')
        plt.plot(np.arange(self.max_classifiers), test_acc, color='blue')
        plt.show()

    def predict(self, x):
        prob = 0
        sum = np.sum(self.classifiers_weight)
        for i in range(len(self.classifiers_weight)):
            prob += self.classifiers[i].predict(x) * self.classifiers_weight[i] /sum # 多个分类器的结果加权平均
        return prob

    def cal_acc(self, x, y):
        y_prob = self.predict(x)
        y_pred = np.where(y_prob> 0.5, 1, 0)
        correct = np.sum(np.where(y_pred == y, 1, 0))
        acc = correct / x.shape[0]
        return acc


# 输入一系列训练好的分类器，画出决策边界的变化
def show_decision_boundary(classifiers, x_train, y_train):
    x, y = creat_data(1000)  # 为了作图显示分类器的决策边界
    fig = plt.figure()

    for num in range(len(classifiers.classifiers)):  # 从1个分类器，到多个分类器
        y_prob = 0
        sum = np.sum(classifiers.classifiers_weight[0:num+1])  # 前num个分类器权重的和
        for i in range(num+1):
            y_prob += classifiers.classifiers[i].predict(x) * classifiers.classifiers_weight[i] / sum

        # 显示预测结果
        plt.subplot(4, 5, num+1)
        for i in range(x.shape[0]):   # 显示决策边界的数据
            if y_prob[i][0] < 0.5:
                plt.scatter(x[i][0], x[i][1], color='pink', linewidths=0.2)
            else:
                plt.scatter(x[i][0], x[i][1], color='lightgreen', linewidths=0.2)

        for i in range(x_train.shape[0]):  # 训练集数据
            if y_train[i][0] < 0.5:
                plt.scatter(x_train[i][0], x_train[i][1], color='red', linewidths=0.1)
            else:
                plt.scatter(x_train[i][0], x_train[i][1], color='green', linewidths=0.1)
    plt.show()


if __name__ == '__main__':
    # 乳腺癌数据集
    # cancer = load_breast_cancer()
    # #x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=42)

    x_train, y_train = creat_data(300)
    x_test, y_test = creat_data(100)

    clf = logistic_regression(2)   # 构建特征数为2的回归模型
    classifiers = adaboost(10, clf, x_train.shape[0])
    classifiers.train(x_train, y_train, x_test, y_test)

    show_decision_boundary(classifiers, x_train, y_train)