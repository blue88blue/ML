from math import log
import operator

def createDataSet():
    dataset = [[0, 0, 0, 0, 'no'],						# 数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataset,labels

# 计算香农熵
def cal_Ent(dataset):
    num = len(dataset)
    lable_counts = {}   # 记录每个类的数量
    for sample in dataset:
        sample_class = sample[-1]
        if sample_class not in list(lable_counts.keys()):
            lable_counts[sample_class] = 0
        lable_counts[sample_class] += 1

    shanonEnt=0
    for key in lable_counts:
        prob = float(lable_counts[key])/num
        shanonEnt -= prob * log(prob, 2)

    return shanonEnt


# feature_index表示使用哪个特征分割数据集，value表示特征的取值
def split_dataset(dataset, feature_index, value):
    sub_dataset = []
    for sample in dataset:
        if sample[feature_index] == value:
            sub_dataset.append(sample.copy())
            del sub_dataset[-1][feature_index]  # 删除当前特征
    return sub_dataset


def choose_best_feature(dataset):
    feature_num = len(dataset[0])-1
    dataset_Ent = cal_Ent(dataset)
    best_feature = 0
    best_info_gain = 0.0
    for i in range(feature_num):  # 按特征的排列顺序，计算各个特征的信息增益
        values = [example[i] for example in dataset]  # 第i个特征的所有取值
        unique_values = set(values)  # 无序不重复集合
        sub_Ent = 0
        for value in unique_values:
            sub_dataset = split_dataset(dataset, i, value)  # 提取出第i个特征等于value的子集
            sub_Ent += len(sub_dataset)/float(len(dataset)) * cal_Ent(sub_dataset)   # 每个子集的经验熵加权求和， 即经验条件熵
        info_gain = dataset_Ent - sub_Ent  # 计算该特征的信息增益
        if info_gain > best_info_gain:   # 取信息增益最大的特征作为最优特征
            best_info_gain = info_gain
            best_feature = i
    return best_feature

# 计算子集中数量最多的类别，并返回类别键值
def majority_count(classlist):
    classes = {}
    for i in classlist:
        if i not in list(classes.keys()):
            classes[i] = 0
        classes[i] += 1
    sorted_classes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)  # 以索引为1的值来进行排序
    return sorted_classes[0][0]


# 递归地构建决策树
def creat_tree(dataset, labels):
    classlist = [sample[-1] for sample in dataset]
    if classlist.count(classlist[0]) == len(dataset):  # 如果类别完全相同，则停止划分，返回该类别。
        return classlist[0]
    if len(dataset) == 1:  # 遍历完所有特征，返回出现次数最多的那个类别。
        return majority_count(classlist)

    best_feature = choose_best_feature(dataset)
    best_feature_label = labels[best_feature]   # 特征的名称，如:年龄

    myTree = {best_feature_label: {}}
    del labels[best_feature]

    feature_values = [example[best_feature] for example in dataset]  # 得到最优特征的所有取值
    feature_values = set(feature_values)
    # 根据最优特征的所有取值划分数据集,递归地构建决策树
    for value in feature_values:
        sublabels = labels[:]  # 拷贝,防止原列表被更改
        # 他的子节点可能是叶节点：特征遍历完或类别全部相同；也可能是一颗子树
        myTree[best_feature_label][value] = creat_tree(split_dataset(dataset, best_feature, value), sublabels)
    return myTree


if __name__ == "__main__":

    dataset ,labels = createDataSet()
    myTree = creat_tree(dataset, labels)
    print(myTree)





