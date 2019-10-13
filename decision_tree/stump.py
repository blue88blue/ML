from math import log
import numpy as np
import operator
import matplotlib.pyplot as plt
import random

class stump:
    def __init__(self):
        self.stump = [0, 0, 0, 0]  # feature_index, value, left_class, right_class

    # 计算香农熵
    def cal_Ent(self, dataset, weight):
        sum = np.sum(weight)
        lable_counts = {}   # 记录每个类的数量
        for i in range(len(dataset)):
            sample_class = dataset[i][-1]
            if sample_class not in list(lable_counts.keys()):
                lable_counts[sample_class] = 0
            lable_counts[sample_class] += weight[i][0]  # 某类的某个样本的权重

        shanonEnt=0
        for key in lable_counts:
            prob = float(lable_counts[key])/sum
            shanonEnt -= prob * log(prob, 2)

        return shanonEnt


    # feature_index表示使用哪个特征分割数据集，value表示特征的取值
    def split_dataset(self, dataset, feature_index, value):
        sub_dataset_l = []
        sub_dataset_r = []
        for sample in dataset:
            if sample[feature_index] > value:
                sub_dataset_r.append(sample.copy())
            else:
                sub_dataset_l.append(sample.copy())
        return sub_dataset_l, sub_dataset_r


    def choose_best_feature(self, dataset, weight):
        feature_num = len(dataset[0])-1
        dataset_Ent = self.cal_Ent(dataset, weight)
        best_feature = 0
        best_value = 0
        best_info_gain = 0.0
        for i in range(feature_num):  # 按特征的排列顺序，计算各个特征的信息增益
            values = [example[i] for example in dataset]  # 第i个特征的所有取值
            unique_values = list(set(values))  # 无序不重复集合
            unique_values.sort()  # 按取值的从小到大排序
            t_a = [(unique_values[i+1]+unique_values[i])/2 for i in range(len(unique_values)-1)]   # 相邻两数的中间值

            for value in t_a:
                sub_dataset_l, sub_dataset_r = self.split_dataset(dataset, i, value)  # 提取出第i个特征等于value的子集
                sub_Ent = len(sub_dataset_l)/float(len(dataset))*self.cal_Ent(sub_dataset_l, weight)\
                          +len(sub_dataset_r)/float(len(dataset))*self.cal_Ent(sub_dataset_r, weight)# 每个子集的经验熵加权求和， 即经验条件熵

                info_gain = dataset_Ent - sub_Ent  # 计算该特征的信息增益
                if info_gain > best_info_gain:   # 取信息增益最大的特征作为最优特征
                    best_info_gain = info_gain
                    best_feature = i
                    best_value = value
        return best_feature, best_value


    # 计算子集中数量最多的类别，并返回类别键值
    def majority_count(self, classlist, weight):
        classes = {}
        for i in range(len(classlist)):
            if classlist[i] not in list(classes.keys()):
                classes[classlist[i]] = 0
            classes[classlist[i]] += weight[i][0]

        sorted_classes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)  # 以索引为1的值来进行排序
        return sorted_classes[0][0]


    def fit(self, dataset, weight):

        best_feature, best_value = self.choose_best_feature(dataset, weight)
        self.stump[0] = best_feature
        self.stump[1] = best_value

        sub_dataset_l, sub_dataset_r = self.split_dataset(dataset, best_feature, best_value)

        classlist_l = [sample[-1] for sample in sub_dataset_l]
        self.stump[2] = self.majority_count(classlist_l, weight)

        classlist_r = [sample[-1] for sample in sub_dataset_r]
        self.stump[3] = self.majority_count(classlist_r, weight)

    def predict(self, x):
        feature_index = x[:, self.stump[0]].reshape([-1,1])
        pred = np.where(feature_index < self.stump[1], self.stump[2], self.stump[3])

        return pred


def creat_data(num, width=100, hight=100):
    data_x = []
    data_y = []
    for num in range(num):
        i = random.randint(0, 99)
        j = random.randint(0, 99)
        data_x.append([i, j])
        if (i-hight/2-40)**2+(j-width/2-40)**2<6400 and (i-hight/2-40)**2+(j-width/2-40)**2>900 :
        # if (i - hight / 2 - 40) ** 2 + (j - width / 2 - 40) ** 2 < 5000:
            data_y.append([1])
        else:
            data_y.append([-1])
    return np.asarray(data_x)/100, np.asarray(data_y)


# stump  输入一系列训练好的分类器，画出决策边界的变化
def show_decision_boundary(classifiers, x_train, y_train):
    x, y = creat_data(1000)  # 为了作图显示分类器的决策边界
    fig = plt.figure()

    for num in range(len(classifiers.classifiers)):  # 从1个分类器，到多个分类器
        y_prob = 0
        sum = np.sum(classifiers.classifiers_weight[0:num+1])  # 前num个分类器权重的和
        for i in range(num+1):
            y_prob += classifiers.classifiers[i].predict(x) * classifiers.classifiers_weight[i] / sum

        # 显示预测结果
        plt.subplot(2, 3, num+1)
        for i in range(x.shape[0]):   # 显示决策边界的数据
            if y_prob[i][0] < 0:
                plt.scatter(x[i][0], x[i][1], color='pink', linewidths=0.2)
            else:
                plt.scatter(x[i][0], x[i][1], color='lightgreen', linewidths=0.2)

        for i in range(x_train.shape[0]):  # 训练集数据
            if y_train[i][0] < 0:
                plt.scatter(x_train[i][0], x_train[i][1], color='red', linewidths=0.1)
            else:
                plt.scatter(x_train[i][0], x_train[i][1], color='green', linewidths=0.1)
    plt.show()



if __name__ == '__main__':
    # weight=np.ones([500,1])
    x_train, y_train = creat_data(500)
    x_test, y_test = creat_data(200)
    train_dataset = np.column_stack([x_train, y_train])

    stump = stump()
    stump.fit(train_dataset, weight)
    print(stump.stump)

    pred = stump.predict(x_test)


