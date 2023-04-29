import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer# 标签二值化
from sklearn.model_selection import train_test_split # 切割数据

digits = load_digits()# 载入数据
X = digits.data# 数据
y = digits.target# 标签

A4=np.array([2,3,4])
b4=np.array([1,2,3])
ans = np.dot(A4,b4)#dot product
#print(A4.shape)
#print(b4.shape)
xx =b4.shape
b4 = b4.reshape(3, 1)

#rint(ans)
print(b4 * A4)
#array([14, 20])