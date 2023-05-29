# 参考https://zhuanlan.zhihu.com/p/524063692
import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # 调用Slider滑块控件


def set_chinese():  # 使得画图时的标题可以为中文
    import matplotlib
    print("[INFO] matplotlib版本为: %s" % matplotlib.__version__)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False


def gamma_tf(image, gamma):
    image = image / 255.0
    new = np.power(image, gamma)
    return new


def gamma_trans(img, gamma=2, eps=0):  # gamma变换函数
    output = 255.*(((img + eps)/255.)**gamma)
    output = np.uint8(output)
    return output


def update_gamma(val):  # 随着滑块变化更新的gamma值进行更新gamma变化
    gamma = slider1.val  # 接收滑块更新后的gamma值
    output_arr = gamma_trans(image1, gamma=gamma, eps=0.5)
    print("-------\n", output_arr)
    hist_aftergamma = cv2.calcHist([output_arr], [0], None, [256], [0, 256])
    plt.subplot(2, 2, 3)
    plt.title('gamma变换后的图像')
    plt.imshow(output_arr, cmap='gray', vmin=0, vmax=255)
    plt.subplot(2, 2, 4)
    plt.cla()
    plt.title('gamma变换后的图像直方图')
    plt.plot(hist_aftergamma)


# 在每一个类或者函数前要留两行空行，在注释的#后面要有一个空格
if __name__ == '__main__':
    set_chinese()
    a = cv2.imread("/home/edify/Code/Machine_Leaning/DLAM1.jpeg", cv2.IMREAD_UNCHANGED)
    image1 = cv2.split(a)[0]  # 蓝
    image2 = cv2.split(a)[1]  # 绿
    image3 = cv2.split(a)[2]  # 红
    # cv2.imshow("raw_image", a)  # 注意：这里不能有中文！！！！！！
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('输入图像')
    plt.imshow(image1, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 2, 2)
    plt.title('输入图像直方图')
    hist_original = cv2.calcHist([image1], [0], None, [256], [0, 256])  # opencv中计算直方图的函数
    plt.plot(hist_original)

    plt.subplots_adjust(bottom=0.3)
    s1 = plt.axes([0.25, 0.1, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    slider1 = Slider(s1, '参数gamma', valmin=0.0, valmax=2.0, valfmt='%.2f', valinit=1, valstep=0.1)  # 实例化Slider控件的类
    slider1.on_changed(update_gamma)  # 调用监控函数来监控gamma的变化并传到update_gamma函数中
    slider1.reset()
    slider1.set_val(1)
    plt.show()
