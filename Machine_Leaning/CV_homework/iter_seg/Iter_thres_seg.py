import cv2
import cv2 as cv
import numpy as np


def Iterate_Thresh(img, initval, MaxIterTimes=20, thre=1):
    """ 阈值迭代算法
    Args:
        img: 灰度图像
        initval: 初始阈值
        MaxIterTimes: 最大迭代次数，默认20
        thre：临界差值，默认为1
    Return:
        计算出的阈值
    """
    mask1, mask2 = (img > initval), (img <= initval)    # 逻辑运算，结果为0，1
    T1 = np.sum(mask1 * img) / np.sum(mask1)
    T2 = np.sum(mask2 * img) / np.sum(mask2)
    T = (T1 + T2) / 2   # 新的阈值
    # 终止条件
    if abs(T - initval) < thre or MaxIterTimes == 0:
        return T
    return Iterate_Thresh(img, T, MaxIterTimes - 1)


img = cv.imread('DLAM1.jpeg', 0)
# 计算灰度平均值
initthre = np.mean(img)
# 阈值迭代
thresh = Iterate_Thresh(img, initthre, 50)

dst = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)[1]
cv.imshow('1', dst)
cv2.imwrite('result.jpg', dst)
cv.waitKey(0)
cv.destroyAllWindows()