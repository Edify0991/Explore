import cv2 as cv
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

def auto_correlation(img, x, y):
    h = img.shape[0]
    w = img.shape[1]
    img1 = np.zeros((h + x, w + y))
    img1[:h, :w] = img
    sum1 = 0
    sum2 = sum(sum(img ** 2))
    for i in range(h):
        for j in range(w):
            sum1 = sum1 + img1[i][j] * img1[i+x][j+y]

    ans = sum1 / sum2
    return ans


if __name__ == "__main__":
    # 读取图像
    img_cv = cv.imread("texture.jpg", cv.IMREAD_GRAYSCALE)

    # 自相关分析
    D_max = 10
    x = np.linspace(0, D_max, D_max+1, dtype=int)
    y = np.linspace(0, D_max, D_max + 1, dtype=int)
    xy_ans = []
    for i in x:
        for j in y:
            if i ** 2 + j ** 2 <= D_max ** 2:
                xy_ans.append((i, j))
    p_ans = np.zeros((1, D_max + 1))
    p_count = np.zeros((1, D_max + 1))
    for k in xy_ans:
        d = math.ceil(math.sqrt(k[0] ** 2 + k[1] ** 2))
        p = auto_correlation(img_cv, k[0], k[1])
        p_ans[0, d] = p_ans[0, d] + p
        p_count[0, d] = p_count[0, d] + 1
    p_ans = p_ans / p_count
    plt.figure()
    plt.plot(x.reshape(-1), p_ans.reshape(-1), 'r', ls='-')
    plt.show()
