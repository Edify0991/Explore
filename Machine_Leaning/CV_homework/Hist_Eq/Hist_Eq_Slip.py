# 参考https://blog.csdn.net/m0_58058919/article/details/126925564
# 直方图均衡化：遍历图像每个像素的灰度，算出每个灰度的概率（n/MN--n是每个灰度的个数，MN是像素总数），用L-1乘以所得概率得到新的灰度
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/edify/Code/Machine_Leaning/DLAM2.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 5x5


def pix_gray(img_gray):  # 直方图统计
    h = img_gray.shape[0]
    w = img_gray.shape[1]

    gray_level = np.zeros(256)
    gray_level2 = np.zeros(256)

    for i in range(h):
        for j in range(w):
            gray_level[img_gray[i, j]] += 1  # 统计灰度级为img_gray[i,j]的个数

    for i in range(1, 256):
        gray_level2[i] = gray_level2[i - 1] + gray_level[i]  # 统计灰度级小于img_gray[i,j]的个数

    return gray_level2


def hist_gray(img_gary):  # 直方图均衡化
    h, w = img_gary.shape
    gray_level2 = pix_gray(img_gray)
    lut = np.zeros(256)
    for i in range(256):  # 实际是(0,255)，不含256
        lut[i] = 255.0 / (h * w) * gray_level2[i]  # 得到新的灰度级
    lut = np.uint8(lut + 0.5)
    out = cv2.LUT(img_gray, lut)
    return out


cv2.imshow('imput', img_gray)
out_img = hist_gray(img_gray)
out_img2 = cv2.equalizeHist(img_gray)
out_img2 = cv2.cvtColor(out_img2, cv2.COLOR_BGR2RGB)
# plt.imshow(out_img2)  # opencv和matplot不能同时使用
# plt.hist(out_img.ravel(), 256)
# plt.show()
cv2.imshow('output', out_img)
cv2.imwrite('out_result.png', out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()