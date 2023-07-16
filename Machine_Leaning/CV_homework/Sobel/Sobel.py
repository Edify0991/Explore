# 参考https://blog.csdn.net/qq_44246412/article/details/115425433
import  numpy as np
import time
import cv2


def convertu8(num):
    if num > 255 or num < -255:
        return 255
    elif -255 <= num <= 255:
        if abs(num - int(num)) < 0.5:
            return np.uint8(abs(num))
        else:
            return np.uint8(abs(num)) + 1


def sobel(img, k=0):
    row = img.shape[0]
    col = img.shape[1]
    image = np.zeros((row, col), np.uint8)
    s = time.time()
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            y = int(img[i - 1, j + 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                        int(img[i, j + 1, k]) - int(img[i, j - 1, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i + 1, j - 1, k])
            x = int(img[i + 1, j - 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                        int(img[i + 1, j, k]) - int(img[i - 1, j, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i - 1, j + 1, k])
            image[i, j] = convertu8(abs(x) * 0.5 + abs(y) * 0.5)
    e = time.time()
    print(e - s)
    return image


if __name__ == '__main__':
    ori_img = cv2.imread("Sobel.jpg")
    sobelimage = sobel(ori_img, 0)
    cv2.imshow("my Result", sobelimage)
    cv2.imwrite('sobel_result.jpg', sobelimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
