import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def gamma_tf(image, gamma):
    image = image / 255.0
    New = np.power(image, gamma)
    return New


# 在每一个类或者函数前要留两行空行，在注释的#后面要有一个空格
if __name__ == '__main__':
    a = cv2.imread("/home/edify/Code/Machine_Leaning/example.jpeg", cv2.IMREAD_UNCHANGED)
    image1 = cv2.split(a)[0]  # 蓝
    image2 = cv2.split(a)[1]  # 绿
    image3 = cv2.split(a)[2]  # 红
    cv2.imshow("原图", a)
    image_1 = gamma_tf(image1, 1.5)
    image_2 = gamma_tf(image1, 0.5)
    merged = cv2.merge([image1, image2, image3])
    cv2.imshow("增强后图1", image_1)
    cv2.imshow("增强后图2", image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
