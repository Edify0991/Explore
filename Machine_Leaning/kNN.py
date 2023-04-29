import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
# iris数据集也称为鸢尾花卉数据集，包含150个数据样本，分三类，每类50个数据，每个数据包含四个属性
iris = datasets.load_iris()
x = iris.data
# 转换成1列
y = iris.target.reshape(-1, 1)
print(x.shape, y.shape)

def distance_Euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis = 1))
def distance_Manhattan(a, b):
    return np.sum(np.abs(a - b), axis = 1)
class kNN():
    def __init__(self, n_neighbors = 1, dist_func = distance_Manhattan):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func
    def fit(self, x, y):
        self.x = x
        self.y = y
    def predict(self, x):
        # shape[0]为矩阵第一维长度
        y_pred = np.zeros((x.shape[0], 1), dtype = self.y.dtype)
        # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值
        for i, x_test in enumerate(x):
            distances = self.dist_func(self.x, x_test)
            n_index = np.argsort(distances)
            # 展开成一维数组
            n_y = self.y[n_index[:self.n_neighbors]].ravel()
            # numpy.bincount函数是统计列表中元素出现的个数
            y_pred[i] = np.argmax(np.bincount(n_y))
            print(np.bincount(n_y))
        # 将数组重新组成一列
        y_pred.reshape(-1)
        return y_pred
# train_test_split方法能够将数据集按照用户的需要指定划分为训练集和测试集
# test_size若在0~1之间，为测试集样本数目与原始样本数目之比；若为整数，则是测试集样本的数目
# random_state为随机数种子，不同的随机数种子划分的结果不同
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
knn = kNN(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print('Prediction:', Y_pred)
print(Y_pred.shape)
print('Test value:', Y_test)
num_correct = np.sum(Y_pred == Y_test)
accuracy = float(num_correct) / X_test.shape[0]
#print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))

knn2 = kNN(n_neighbors = 3, dist_func = distance_Manhattan)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print('Prediction:', Y_pred)
print(Y_pred.shape)
print('Test value:', Y_test)
num_correct = np.sum(Y_pred == Y_test)
accuracy = float(num_correct) / X_test.shape[0]
print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))

knn3 = kNN()
knn3.fit(X_train, Y_train)
result_list = []
for k in range(1, 50, 5):
    knn3.n_neighbors = k
    print(knn3.n_neighbors)
    knn3.dist_func = distance_Euclidean
    Y_pred = knn3.predict(X_test)
    num_correct = np.sum(Y_pred == Y_test)
    accuracy = float(num_correct) / X_test.shape[0]
    result_list.append([k, accuracy])
ans = pd.DataFrame(result_list, columns = ['k', '预测准确率'])
print(ans)