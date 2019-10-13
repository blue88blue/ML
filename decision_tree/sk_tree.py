from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
from sklearn.datasets import load_breast_cancer
import pydot


cancer = load_breast_cancer()
iris = load_iris()

# 参数random_state是指随机生成器，0表示函数输出是固定不变的
x_train,x_test,y_train,y_test = train_test_split(cancer['data'],cancer['target'],test_size=0.3,random_state=42)

tree = DecisionTreeClassifier(max_depth=4,random_state=0)
tree.fit(x_train,y_train)

train_acc = tree.score(x_train,y_train)
test_acc = tree.score(x_test,y_test)
print("train_acc:%f"%train_acc)
print("test_acc:%f"%test_acc)

#生成可视化图
export_graphviz(tree,out_file="tree.dot",class_names=['setosa', 'versicolor', 'virginica'],feature_names=cancer.feature_names,impurity=True, filled=True)
#展示可视化图
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
