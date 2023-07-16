# 参考https://blog.csdn.net/sgzqc/article/details/119682864
import cv2
import numpy as np
from queue import Queue
import math


def get_binary_img(img):
    # gray img to bin image
    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j] = 255 if img[i][j] > 127 else 0
    return bin_img


def mode1():
    gray_img = np.array([[1, 3, 4, 7, 6],
                         [2, 1, 6, 8, 6],
                         [4, 7, 8, 6, 7],
                         [5, 8, 9, 6, 7],
                         [4, 2, 7, 6, 1]])
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    out_img = np.zeros(shape=gray_img.shape, dtype=np.uint8)
    seeds = [(2, 3)]
    seeds = np.array(seeds)
    ins = label = 1
    flag = np.zeros(shape=gray_img.shape, dtype=np.uint8)
    for seed in seeds:
        x = seed[0]
        y = seed[1]
        out_img[y][x] = 255
        flag[seed[1]][seed[0]] = label
    # 8 邻域
    directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    visited = np.zeros(shape=gray_img.shape, dtype=np.uint8)
    thre = 2
    queue_obj = Queue()  # 创建一个队列对象
    queue_obj.put(seeds)
    sum = 0  # 用来统计同一label的像素的灰度和
    n = 0  # 用来统计同一label的像素个数
    while not queue_obj.empty():
        seed = queue_obj.get()
        seed = seed.flatten()
        x = seed[0]
        y = seed[1]
        print(x, y, flag[y][x])
        # visit point (x,y)
        visited[y][x] = 1
        if flag[y][x] == 1:  # 第一个种子点，取出该像素后，队列为空
            mean = gray_img[y][x]
        elif flag[y][x] != ins:
            sum = gray_img[y][x]  # 因为队列首部像素已经弹出，所以这里需要单独相加
            n = 1
            ins = flag[y][x]
            for i in list(queue_obj.queue):
                if flag[i[1]][i[0]] == flag[y][x]:
                    sum += gray_img[i[1]][i[0]]
                    n += 1
            mean = sum / n
            print(out_img, flag[y][x])
        low_thre = mean - thre
        high_thre = mean + thre
        print('low', low_thre, 'high', high_thre)
        for direct in directs:
            cur_x = x + direct[0]
            cur_y = y + direct[1]
            # 非法
            if cur_x < 0 or cur_y < 0 or cur_x >= w or cur_y >= h:
                continue
            # 没有访问过
            if (not visited[cur_y][cur_x]) and high_thre >= abs(gray_img[cur_y][cur_x]) >= low_thre:
                out_img[cur_y][cur_x] = 255
                visited[cur_y][cur_x] = 1
                queue_obj.put(np.array([cur_x, cur_y]))
                flag[cur_y][cur_x] = flag[y][x] + 1
    print(gray_img)
    print(out_img)


def mode2(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 调用，二值化图像
    bin_img = get_binary_img(gray_img)
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    out_img = np.zeros(shape=bin_img.shape, dtype=np.uint8)
    # 选取初始种子点,选择初始3个种子点
    seeds = [(176, 255), (229, 405), (347, 165)]
    # 8 邻域
    directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    visited = np.zeros(shape=bin_img.shape, dtype=np.uint8)
    while len(seeds):
        seed = seeds.pop(0)
        x = seed[0]
        y = seed[1]
        # visit point (x,y)
        visited[y][x] = 1
        for direct in directs:
            cur_x = x + direct[0]
            cur_y = y + direct[1]
            # 非法
            if cur_x < 0 or cur_y < 0 or cur_x >= w or cur_y >=h:
                continue
            # 没有访问过且属于同一目标
            if (not visited[cur_y][cur_x]) and (bin_img[cur_y][cur_x] == bin_img[y][x]):
                out_img[cur_y][cur_x] = 255
                visited[cur_y][cur_x] = 1
                seeds.append((cur_x, cur_y))
    bake_img = img.copy()
    h = bake_img.shape[0]
    w = bake_img.shape[1]
    for i in range(h):
        for j in range(w):
            if out_img[i][j] != 255:
                bake_img[i][j][0] = 0
                bake_img[i][j][1] = 0
                bake_img[i][j][2] = 0
    cv2.imshow('bake_img', bake_img)
    cv2.imwrite('reg_grow_out.jpg', bake_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_name = "/home/edify/Code/Machine_Leaning/reg_grow.jpg"
    img = cv2.imread(img_name)
    print("mode1:课件样例，区域生长条件要考虑邻域与种子像素灰度值大小关系")
    mode1()
    print("mode2:可以有多个种子，对象是二值化图像，因此区域生长条件为领域与种子像素灰度值大小相等，结果如图片所示")
    mode2(img)