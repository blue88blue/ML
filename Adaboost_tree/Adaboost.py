from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
from sklearn.datasets import load_breast_cancer
import pydot
from copy import deepcopy
import matplotlib.pyplot as plt




class adaboost_tree:
    def __init__(self, max_classifiers):
        self.max_classifiers = max_classifiers
        self.classifiers = []
        self.classifiers_weight = []

    def cal_error(self, tree, samples_weight, x, y):
        y_porb = tree.predict_proba(x)  # 测试
        y_pred = np.where(y_porb[:, 0] > 0.5, 0, 1)
        misclassified_samples = np.where(y_pred == y, 0, 1)  # 分类错误的样本记为1
        error_rate = np.sum(samples_weight * misclassified_samples) / np.sum(samples_weight)
        return error_rate, misclassified_samples

    def update_samples_weight(self, samples_weight, misclassified_samples, error_rate):
        d = np.sqrt((1-error_rate)/error_rate)
        self.classifiers_weight.append(np.log(d))  # 分类器的权重

        scale = np.where(misclassified_samples == 1, d, 1/d)  # 若分类错误的样本权重×d,分类正确的样本权重/d
        samples_weight = samples_weight*scale

        return samples_weight


    def train(self,x_train,y_train, x_test, y_test):
        images = []
        train_acc = []
        test_acc = []

        samples_weight = np.ones(x_train.shape[0])  # 初始化权重为1

        for i in range(self.max_classifiers):
            tree = DecisionTreeClassifier(max_depth=4, random_state=i)
            tree.fit(x_train, y_train, sample_weight=samples_weight)  # 根据样本权重训练
            self.classifiers.append(deepcopy(tree))  # 得到一个分类器。深度拷贝，拷贝当前对象及其所有子对象

            error_rate, misclassified_samples = self.cal_error(tree, samples_weight, x_train, y_train)  # 计算训练错误率
            samples_weight = self.update_samples_weight(samples_weight, misclassified_samples, error_rate)  # 更新样本权重

            train_acc.append(self.cal_acc(x_train, y_train))
            test_acc.append(self.cal_acc(x_test, y_test))
            print("train acc:%f  " %(train_acc[i]))

            self.show_img(images, x_train)

        self.plot(images)

        plt.xlabel('iterations')
        plt.ylabel('acc')
        plt.plot(np.arange(self.max_classifiers), train_acc, color='green')
        plt.plot(np.arange(self.max_classifiers), test_acc, color='blue')
        plt.show()

    def predict(self, x):
        prob = 0
        sum = np.sum(self.classifiers_weight)
        for i in range(len(self.classifiers)):
            prob += np.asarray(self.classifiers[i].predict_proba(x)) * self.classifiers_weight[i] / sum  # 多个分类器的结果加权平均
        return prob

    def cal_acc(self, x, y):
        y_prob = self.predict(x)
        y_pred = np.where(y_prob[:, 0] > 0, 0, 1)
        correct = np.sum(np.where(y_pred == y, 1, 0))
        acc = correct / x.shape[0]
        return acc

    def tree_visualization(self):
        for i in range(len(self.classifiers)):
            # 生成可视化图
            export_graphviz(self.classifiers[i], out_file="output/tree/tree_"+str(i)+".dot", class_names=['严重', '轻微'],
                            impurity=True, filled=True)
            # 展示可视化图
            (graph,) = pydot.graph_from_dot_file("output/tree/tree_"+str(i)+".dot")
            graph.write_png("output/tree/tree_"+str(i)+".png")

    def show_img(self, images, x, width=150, hight=150):
        y_prob = self.predict(x)
        y_pred = np.where(y_prob[:, 0] > 0.5, 0, 1)

        img = np.zeros([hight, width])
        for i in range(len(x)):
            img[x[i][0]][x[i][1]] = y_pred[i]
        images.append(img)

    def plot(self, images):
        plt.figure(figsize=(10, 10))  # 设置窗口大小
        for i in range(self.max_classifiers):
            plt.subplot(4, 5, i+1), plt.title('T'+str(i))
            plt.imshow(images[i],cmap='gray'), plt.axis('off')
        plt.show()




if __name__ == '__main__':
    # 乳腺癌数据集
    # cancer = load_breast_cancer()
    data_x, data_y = creat_data()
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=42)

    classifiers = adaboost_tree(20)
    classifiers.train(data_x, data_y, x_test, y_test)
    # classifiers.tree_visualization()



