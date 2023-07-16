import cv2 as cv
import numpy as np
from skimage import data
import matplotlib.pyplot as plt


# Gray scale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out


def zxcor(f, D, m, n):
    # 自相关函数zxcor(), f为读入的图像数据，D为偏移距离，[m, n]是图像的尺寸数据，返回图像相关函数C的值
    # epsilon和eta是自相关函数C的偏移变量
    f2 = np.zeros((D, D))
    f3 = np.zeros((D, D))
    C = np.zeros((D, D))
    for epsilon in range(D - 1):
        for eta in range(D - 1):
            temp = 0
            fp = 0
            for x in range(m - 1):
                for y in range(n - 1):
                    # print(x, epsilon, y, eta, m, n)
                    if ((x + epsilon) >= m) | ((y + eta) >= n):
                        f1 = 0
                    else:
                        # print(y+eta, n)
                        f1 = f[x][y] * f[x + epsilon][y + eta]
                    temp = f1 + temp
                    fp = f[x][y] * f[x][y] + fp
            f2[epsilon][eta] = temp
            f3[epsilon][eta] = fp
            C[epsilon][eta] = f2[epsilon][eta] / f3[epsilon][eta]
    epsilon = np.arange(D)
    eta = np.arange(D)
    return epsilon, eta, C


# img = cv.imread('scenary.jpeg').astype(np.float32)
img = data.grass()
# gray = BGR2GRAY(img)
gray = img.copy()
height = gray.shape[0]
width = gray.shape[1]
epsilon, eta, C = zxcor(gray, D=10, m=height, n=width)
# 生成网格点坐标矩阵,对x和y数据执行网格化
X, Y = np.meshgrid(epsilon, eta)
# 计算z轴数据
Z = C.copy()
print(C)
# 定义新坐标轴
fig = plt.figure()
ax3 = plt.axes(projection='3d')
# 绘图
# 函数plot_surface期望其输入结构为一个规则的二维网格
ax3.plot_surface(X, Y, C)  # cmap是颜色映射表
plt.title("3D")
plt.show()
print(C.shape)