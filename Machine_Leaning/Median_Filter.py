# 参考https://zhuanlan.zhihu.com/p/523659722
import cv2
import numpy as np
from PIL import Image

cv2.waitKey(0)


# 添加椒盐噪声
def AddNoise(imarray, probility=0.05, method="salt_pepper"):  # 灰度图像
    # 获取图片的长和宽
    height, width = imarray.shape[:2]

    for i in range(height):
        for j in range(width):
            if np.random.random(1) < probility:  # 随机加盐或者加椒
                if np.random.random(1) < 0.5:
                    imarray[i, j] = 0
                else:
                    imarray[i, j] = 255
    return imarray


# 中值滤波法
def medianBlur(image, ksize=7):
    '''
    中值滤波，去除椒盐噪声

    args:
        image：输入图片数据,要求为灰度图片
        ksize：滤波窗口大小
    return：
        中值滤波之后的图片
    '''
    rows, cols = image.shape[:2]
    # 输入校验
    half = ksize // 2   # 整除
    startSearchRow = half
    endSearchRow = rows - half
    startSearchCol = half
    endSearchCol = cols - half
    dst = np.zeros((rows, cols), dtype=np.uint8)
    # 中值滤波
    for y in range(startSearchRow, endSearchRow):
        for x in range(startSearchCol, endSearchCol):
            window = []
            if ksize % 2 == 1:
                for i in range(y - half, y + half + 1):
                    for j in range(x - half, x + half + 1):
                        window.append(image[i][j])
            else:
                for i in range(y - half, y + half):
                    for j in range(x - half, x + half):
                        window.append(image[i][j])
            # 取中间值
            window = np.sort(window, axis=None)

            if len(window) % 2 == 1:
                medianValue = window[len(window) // 2]
            else:
                medianValue = int(window[len(window) // 2 - 1] / 2 + window[len(window) // 2] / 2)
            dst[y][x] = medianValue

    return dst


# 自适应中值滤波法
def amp_medianBlur(img, ksize=2):
    # 图像边缘扩展
    # 为保证边缘的像素点可以被采集到，必须对原图进行像素扩展。
    # 一般设置的最大滤波窗口为7，所以只需要向上下左右各扩展3个像素即可采集到边缘像素。
    m, n = img.shape[:2]
    Nmax = 3
    imgn = np.zeros((m + 2 * Nmax, n + 2 * Nmax),dtype=np.uint8)

    imgn[ Nmax : (m + Nmax ) , Nmax : (n + Nmax ) ] = img[:,:,0].copy() # 将原图覆盖在imgn的正中间

    # 下面开始向外扩展，即把边缘的像素向外复制
    imgn[0: Nmax, Nmax: n + Nmax]=img[0: Nmax, 0 : n,0].copy() # 扩展上边界
    imgn[0: m + Nmax, n + Nmax : n + 2 * Nmax]=imgn[0: m+Nmax, n: n + Nmax].copy() # 扩展右边界
    imgn[m + Nmax: m + 2 * Nmax, Nmax : n + 2 * Nmax]=imgn[m : m + Nmax,Nmax : n + 2 * Nmax].copy() #扩展下边界
    imgn[0: m + 2 * Nmax,0: Nmax]=imgn[0: m + 2 * Nmax,Nmax : 2 * Nmax].copy() # 扩展左边界
    re = imgn.copy()  # 扩展之后的图像

    # 得到不是噪声点的中值
    for i in range(Nmax,m+Nmax+1):
        for j in range(Nmax,n+Nmax+1):
            r = 1 # 初始向外扩张1像素，即滤波窗口大小为3
            while r!=Nmax+1: #当滤波窗口小于等于7时（向外扩张元素小于4像素）

                W = imgn[i - r-1:i + r,j - r-1: j + r].copy()
                Imin,Imax  = np.min(W),np.max(W) # 最小灰度值 # 最大灰度值

                # 取中间值
                window = np.sort(W, axis=None)

                if len(window) % 2 == 1:
                    Imed = window[len(window) // 2]

                else:
                    Imed = int((window[len(window) // 2] + window[len(window) // 2 + 1]) / 2)
                if Imin < Imed and Imed < Imax: # 如果当前窗口中值不是噪声点，那么就用此次的中值为替换值
                    break;
                else:
                    r = r + 1; #否则扩大窗口，继续判断，寻找不是噪声点的中值
                # 判断当前窗口内的中心像素是否为噪声，是就用前面得到的中值替换，否则不替换
            if Imin < imgn[i, j] and imgn[i, j] < Imax:  # 如果当前这个像素不是噪声，原值输出
                re[i, j] = imgn[i, j].copy()
            else:  # 否则输出邻域中值
                re[i, j] = Imed
    return re


# 读取图片
image = cv2.imread("Lena.jpg")
width = image.shape[0]
height = image.shape[1]
grayimg = np.zeros([width, height, 1], np.uint8)
for i in range(height):
    for j in range(width):
        grayimg[i, j] = 0.299 * image[i, j][0] + 0.587 * image[i,j ][1] + 0.114 * image[i, j][2]
cv2.imshow('srcImage', image)
cv2.imshow('grayImage', grayimg)

img_addnoise = AddNoise(grayimg)
cv2.imshow('addnoise_Image', img_addnoise)
# remig2 = amp_medianBlur(img_addnoise)
# cv2.imshow('re_Image2', remig2)
reimg = medianBlur(img_addnoise)
cv2.imshow('re_Image', reimg)

cv2.waitKey(0)