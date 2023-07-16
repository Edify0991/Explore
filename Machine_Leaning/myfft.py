import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt


# 自定义傅里叶变换功能函数
def dft(img):
    # 获取图像属性
    # H, W , channel = img.shape
    H, W = img.shape
    # 定义频域图，从公式可以看出为求出结果为复数，因此，需要定义为复数矩阵
    # F = np.zeros((H, W,channel), dtype=complex)
    F = np.zeros((H, W), dtype=complex)
    # 准备与原始图像位置相对应的处理索引
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)
    # 通过公式遍历
    # for c in range(channel):  # 对彩色的3通道数进行遍历
    for u in range(H):
        for v in range(W):
            print(u, v)
            # F[u, v, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * u / W + y * v / H))) / np.sqrt(H * W)
            F[u, v] = np.sum(img[...] * np.exp(-2j * np.pi * (x * u / W + y * v / H))) / np.sqrt(H * W)
    print('hhh')
    return F


if __name__ == '__main__':
    # gray = cv2.imread("/home/edify/Code/Machine_Leaning/Lena.jpg", 1)
    img = data.coffee()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))
    img_dft = dft(gray)
    dft_shift = np.fft.fftshift(img_dft)
    fimg = np.log(np.abs(dft_shift))
    # cv2.imshow("fimg", fimg)
    # cv2.imshow("gray", gray)
    plt.subplot(131), plt.imshow(gray, 'gray'), plt.title('原图像')
    plt.axis('off')
    plt.subplot(132), plt.imshow(np.int8(fimg), 'gray'), plt.title('傅里叶变换')
    plt.axis('off')
    img_dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(img_dft)
    fimg = 20 * np.log(cv2.magnitude(dft_shift[:, :, 1], dft_shift[:, :, 0]))
    plt.subplot(133), plt.imshow(np.int8(fimg), 'gray'), plt.title('傅里叶xinde变换')
    plt.axis('off')
    plt.show()
    # cv2.imshow("dft", fimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
