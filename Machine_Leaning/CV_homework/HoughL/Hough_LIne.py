import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math


class Hough(object):
    def __init__(self, n_edges, theta_min=-90, theta_max=90, pix_threshold=10):
        self.thetas = [i for i in range(theta_min, theta_max+1)] # angles
        self.n_edges = n_edges # max n_edges to draw
        self.pix_threshold = pix_threshold # consider pixels whose value is over this

    def vote(self, img):
        self.rmax = int(math.hypot(img.shape[0], img.shape[1]))  # the maximum value rho can get. 欧几里德范数
        self.hough_space = np.zeros((len(self.thetas), 2 * self.rmax + 1))  # This is the hough space that we will vote.

        for x in range(img.shape[1]):  # the X and Y coordinates in an image is different thats why x == img.shape[1]
            for y in range(img.shape[0]):
                if img[y, x] > self.pix_threshold:
                    for i, theta in enumerate(self.thetas):  # rotate the line
                        th = math.radians(theta)
                        ro = round(x * math.cos(th) + y * math.sin(th)) + self.rmax  # we add r_max to get rid of negative values for indexing.
                        if ro <= 2 * self.rmax:
                            self.hough_space[i, ro] += 1  # vote

    def get_lines(self):  # This method simply returns top n_edges in the hough space.
        return self.topk(self.hough_space, k=self.n_edges)

    def topk(self, a, k):
        idx = np.argpartition(a.ravel(), a.size - k)[-k:]
        return np.column_stack(np.unravel_index(idx, a.shape))  # 将一维数组作为列堆叠到二维数组中

    def reset(self):  # reset if needed
        self.hough_space = np.zeros((len(self.thetas), 2 * self.rmax + 1))


class CannyMethod(object):
    def __init__(self, max_lowThreshold=100, ratio=3, kernel_size=3):
        self.max_lowThreshold = max_lowThreshold # hysterisis thresholding
        self.ratio = ratio
        self.kernel_size = kernel_size # gaussian filter size

    def getEdgeMap(self, src, src_gray, threshold):
        low_threshold = threshold
        img_blur = cv.blur(src_gray, (3, 3))  # clear noise
        detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * self.ratio, self.kernel_size)  # get edge map
        mask = detected_edges != 0
        dst = src * (mask[:, :, None].astype(src.dtype))
        return detected_edges, dst  # return edge_map along with original image


class ExtractLines(object):
    def __init__(self,seg_path,src_path):
        self.Hough = Hough(20)
        self.Canny = CannyMethod()
        self.seg_path = seg_path
        self.src_path = src_path

    def start(self):
        file = '/query.jpg'
        orig_image = cv.imread(self.src_path + file)
        seg_image = cv.imread(self.seg_path + file)
        src_gray = cv.cvtColor(orig_image, cv.COLOR_BGR2GRAY)
        img, rgbimg = self.Canny.getEdgeMap(orig_image, src_gray, 100)
        self.Hough.vote(img)
        lines = self.Hough.get_lines()
        f = plt.figure(figsize=(20, 10))
        f.add_subplot(1, 4, 1)
        plt.imshow(orig_image, interpolation='nearest')
        plt.axis("off")
        f.add_subplot(1, 4, 2)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis("off")
        # plt.savefig(fname="hello.png")
        lined_orig, lined_seg = self.draw_lines(lines, orig_image, seg_image)
        f.add_subplot(1, 4, 3)
        plt.imshow(lined_orig, interpolation='nearest')
        plt.axis("off")
        f.add_subplot(1, 4, 4)
        plt.imshow(lined_seg, interpolation='nearest')
        plt.axis("off")
        plt.show()
        self.Hough.reset()

    def draw_lines(self, lines, orig, seg):
        for line in lines:
            th = math.radians(self.Hough.thetas[line[0]])
            ro = line[1] - self.Hough.rmax  # we extract rmax since we added it on the preprocessing phase.
            a = math.cos(th)
            b = math.sin(th)
            x0 = a * ro
            y0 = b * ro
            x1 = int(round(x0 + 1000 * (-b)))  # 根据OX0 dot X0X1 = 0 两向量点积为0，垂直关系
            y1 = int(round(y0 + 1000 * (a)))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * (a)))
            cv.line(orig, (x1, y1), (x2, y2), (255, 0, 0), 3, 8)
            cv.line(seg, (x1, y1), (x2, y2), (255, 0, 0), 3, 8)
        return orig, seg


if __name__ == "__main__":
    seg_path = '/home/edify/Code/Machine_Leaning'
    source_path = '/home/edify/Code/Machine_Leaning'
    Extractor = ExtractLines(seg_path, source_path)
    Extractor.start()