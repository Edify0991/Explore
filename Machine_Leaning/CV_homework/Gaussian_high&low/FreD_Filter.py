# 参考1：https://blog.csdn.net/qq_38463737/article/details/118682500
# 参考2：https://zhuanlan.zhihu.com/p/387352802
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 理想低通滤波器
def LowPassFilter(img):
    """
    理想低通滤波器
    """
    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置低通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 20:crow + 20, ccol - 20:ccol + 20] = 1

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    return res


# 理想高通滤波器
def HighPassFilter(img):
    """
    理想高通滤波器
    """
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 设置高通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - 2:crow + 2, ccol - 2:ccol + 2] = 0

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg


# 巴特沃斯低通滤波器
def ButterworthLowPassFilter(image, d, n, s1):
    """
    Butterworth低通滤波器
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):

                def cal_distance(pa, pb):
                    from math import sqrt

                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis
                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 / (1 + (dis / d) ** (2 * n))
        return transform_matrix


    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


# 巴特沃斯高通滤波器
def ButterworthHighPassFilter(image, d, n, s1):
    """
    Butterworth高通滤波器
    d : 设置半径
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):

                def cal_distance(pa, pb):
                    from math import sqrt

                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 / (1 + (math.sqrt(2) - 1) * (d / dis) ** (2 * n))
        return transform_matrix


    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix))) # 将图移回去
    return new_img


# 高斯低通滤波器
def GaussianLowPassFilter(image, sigma, s1):
    """
    Gaussian低通滤波器
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix():
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):

                def cal_distance(pa, pb):
                    from math import sqrt

                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis
                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = np.exp(- (dis ** 2) / (2 * (sigma ** 2)))
        return transform_matrix


    d_matrix = make_transform_matrix()
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img

# 高斯高通滤波器
def GaussianHighPassFilter(image, sigma, s1):
    """
    Gaussian高通滤波器
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix():
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):

                def cal_distance(pa, pb):
                    from math import sqrt

                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis
                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 - np.exp(- (dis ** 2) / (2 * (sigma ** 2)))
        return transform_matrix


    d_matrix = make_transform_matrix()
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img

# 指数滤波器
def filter(img, D0, W=None, N=2, type='lp', filter='exponential'):
    '''
    频域滤波器
    Args:
        img: 灰度图片
        D0: 截止频率
        W: 带宽
        N: butterworth和指数滤波器的阶数
        type: lp, hp, bp, bs即低通、高通、带通、带阻
    Returns:
        imgback：滤波后的图像
    '''

    # 离散傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 中心化
    dtf_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.ones((rows, cols, 2))  # 生成rows行cols列的2纬矩阵
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if (filter.lower() == 'exponential'):  # 指数滤波器
                if (type == 'lp'):
                    mask[i, j] = np.exp(-(D / D0) ** (2 * N))
                elif (type == 'hp'):
                    mask[i, j] = np.exp(-(D0 / D) ** (2 * N))
                elif (type == 'bs'):
                    mask[i, j] = np.exp(-(D * W / (D ** 2 - D0 ** 2)) ** (2 * N))
                elif (type == 'bp'):
                    mask[i, j] = np.exp(-((D ** 2 - D0 ** 2) / D * W) ** (2 * N))
                else:
                    assert ('type error')

    fshift = dtf_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 计算像素梯度的绝对值
    img_back = np.abs(img_back)
    img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))

    return img_back


def put(path):
    img = cv2.imread(path, 1)
    # img = cv2.imread(os.path.join(base, path), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # 将FFT输出中的直流分量移动到频谱中央
    # 取绝对值后将复数变化为实数
    # 取对数的目的是将数据变换到0~255
    s1 = np.log(np.abs(fshift))

    # 用以中文显示
    plt.subplot(4, 3, 1)
    plt.axis('off')
    plt.title('原始图像')
    plt.imshow(img, cmap='gray')

    plt.subplot(4, 3, 2)
    plt.axis('off')
    plt.title('理想低通20')
    res1 = LowPassFilter(img)
    plt.imshow(res1, cmap='gray')

    plt.subplot(4, 3, 3)
    plt.axis('off')
    plt.title('理想高通2')
    res2 = HighPassFilter(img)
    plt.imshow(res2, cmap='gray')

    plt.subplot(4, 3, 4)
    plt.axis('off')
    plt.title('原始图像')
    plt.imshow(img, cmap='gray')

    plt.subplot(4, 3, 5)
    plt.axis('off')
    plt.title('巴特沃斯低通20')
    butter_10_1 = ButterworthLowPassFilter(img, 20, 1, s1)
    plt.imshow(butter_10_1, cmap='gray')

    plt.subplot(4, 3, 6)
    plt.axis('off')
    plt.title('巴特沃斯高通2')
    butter_2_1_1 = ButterworthHighPassFilter(img, 2, 1, s1)
    plt.imshow(butter_2_1_1, cmap='gray')

    plt.subplot(4, 3, 7)
    plt.axis('off')
    plt.title('指数原始图像')
    plt.imshow(img, cmap='gray')

    plt.subplot(4, 3, 8)
    plt.axis('off')
    plt.title('指数低通图像20')
    img_back = filter(img, 30, type='lp')
    plt.imshow(img_back, cmap='gray')

    plt.subplot(4, 3, 9)
    plt.axis('off')
    plt.title('指数高通图像2')
    img_back = filter(img, 2, type='hp')
    plt.imshow(img_back, cmap='gray')

    plt.subplot(4, 3, 10)
    plt.axis('off')
    plt.title('高斯原始图像')
    plt.imshow(img, cmap='gray')

    plt.subplot(4, 3, 11)
    plt.axis('off')
    plt.title('高斯低通图像')
    img_back = GaussianLowPassFilter(img, 20, s1)
    plt.imshow(img_back, cmap='gray')

    plt.subplot(4, 3, 12)
    plt.axis('off')
    plt.title('高斯高通图像')
    img_back = GaussianHighPassFilter(img, 15, s1)
    plt.imshow(img_back, cmap='gray')

    # plt.savefig('2.new.jpg')
    plt.show()


# 处理函数，要传入路径
put('Lena.jpg')
