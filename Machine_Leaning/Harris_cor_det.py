# 参考https://blog.csdn.net/weixin_46585836/article/details/128129608
import cv2
import numpy as np


def My_corner_Harris(image, blockSize, ksize, k):
    # 将图片转化为灰度图像，并转化类型为float32
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src = gray_img.astype(np.float32)

    # 获取图像长和高
    SrcHeight = src.shape[0]
    SrcWidth = src.shape[1]

    # 利用Sobel函数计算图像梯度
    # src为原图像，-1表示输出图像大小与原图像相同
    # ksize为sobel算子，定义为角点检测敏感度，必须为3-31之间的奇数
    # (1,0)表示对x求偏导，(0,1)表示对y求偏导
    Ix = cv2.Sobel(src, -1, 1, 0, ksize)
    Iy = cv2.Sobel(src, -1, 0, 1, ksize)

    # 计算Ix2, Ixy, Iy2
    Ix2 = np.multiply(Ix, Ix)
    Ixy = np.multiply(Ix, Iy)
    Iy2 = np.multiply(Iy, Iy)

    # 使用高斯平滑滤波进行加权计算
    Ix2 = cv2.GaussianBlur(Ix2, (blockSize, blockSize), 1.3)
    Ixy = cv2.GaussianBlur(Ixy, (blockSize, blockSize), 1.3)
    Iy2 = cv2.GaussianBlur(Iy2, (blockSize, blockSize), 1.3)

    # 计算最后的R值
    R = np.zeros((SrcHeight, SrcWidth))  # 定义空的R矩阵
    for i in range(SrcHeight):
        for j in range(SrcWidth):
            M = np.array([[Ix2[i, j], Ixy[i, j]], [Ixy[i, j], Iy2[i, j]]])
            R[i, j] = np.linalg.det(M) - k * ((M.trace()) ** 2)
    return R


# detector parameters
block_size = 5
sobel_size = 3
k = 0.04

image = cv2.imread('scenary.jpeg')
R = My_corner_Harris(image, block_size, sobel_size, k)
image[R > 0.01 * R.max()] = [0, 0, 255]

cv2.imshow('detection result', image)

cv2.waitKey(0)
cv2.destroyAllWindows()