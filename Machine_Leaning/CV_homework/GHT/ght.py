# 参考https://blog.csdn.net/wi162yyxq/article/details/112562286
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import imageio.v3 as iio
from skimage.feature import canny
from scipy.ndimage import sobel

# Good for the b/w test images used
MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50


def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    gradient = np.arctan2(dy, dx) * 180 / np.pi

    return gradient


def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    edges = canny(image, low_threshold=MIN_CANNY_THRESHOLD,
                  high_threshold=MAX_CANNY_THRESHOLD)  # 建立R-Tabel需要在模板边缘进行
    gradient = gradient_orientation(edges)  # 计算边缘各点的梯度角度

    r_table = defaultdict(list)  # defaultdict可以产生一个带有默认值的dict，如果key不存在，就会返回默认值
    for (i, j), value in np.ndenumerate(edges):
        if value:
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table


def accumulate_gradients(r_table, grayImage):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD,
                  high_threshold=MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)

    accumulator = np.zeros(grayImage.shape)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            for r in r_table[gradient[i][j]]:  # 遍历R-Talbe中每一行(每行为一个梯度角)
                accum_i, accum_j = i + r[0], j + r[1]  # 行中每个元素为模板中边缘点与参考点的向量，这一步可计算出测试图像中某个边缘点的所有理想参考点并进行投票
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                    accumulator[int(accum_i), int(accum_j)] += 1

    return accumulator


def general_hough_closure(reference_image):
    '''
    Generator function to create a closure with the reference image and origin
    at the center of the reference image

    Returns a function f, which takes a query image and returns the accumulator
    '''
    referencePoint = (reference_image.shape[0] / 2, reference_image.shape[1] / 2)  # 选取模板参考点为图像正中心
    r_table = build_r_table(reference_image, referencePoint)

    def f(query_image):
        return accumulate_gradients(r_table, query_image)

    return f


def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[-n:]  # 为了排序找出前n大的数，先将整个矩阵展开成一维，这里结果为前n大的数的索引
    indices = (np.unravel_index(i, a.shape) for i in indices)  # 找出前n大的数在矩阵中的坐标
    return [(a[i], i) for i in indices]  # 返回坐前n大的票数及在图像中的坐标


def test_general_hough(gh, reference_image, query):
    '''
    Uses a GH closure to detect shapes in an image and create nice output
    '''
    query_image = iio.imread(query)
    img_r = query_image[:, :, 0]
    img_g = query_image[:, :, 1]
    img_b = query_image[:, :, 2]
    query_image_gray = img_r * 0.299 + img_g * 0.587 + img_b * 0.114

    # accumulator = accumulate_gradients(gh, query_image_gray)
    accumulator = gh(query_image_gray)

    plt.clf()
    plt.gray()

    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.title('Reference image')
    plt.imshow(reference_image)

    fig.add_subplot(2, 2, 2)
    plt.title('Query image')
    plt.imshow(query_image_gray)

    fig.add_subplot(2, 2, 3)
    plt.title('Accumulator')
    plt.imshow(accumulator)

    fig.add_subplot(2, 2, 4)
    plt.title('Detection')
    plt.imshow(query_image_gray)

    # top 5 results in red
    m = n_max(accumulator, 5)
    y_points = [pt[1][0] for pt in m]
    x_points = [pt[1][1] for pt in m]
    plt.scatter(x_points, y_points, marker='o', color='r')

    # top result in yellow
    i, j = np.unravel_index(accumulator.argmax(), accumulator.shape)
    plt.scatter([j], [i], marker='x', color='y')

    # d, f = os.path.split(query)[0], os.path.splitext(os.path.split(query)[1])[0]
    # plt.savefig(os.path.join(d, f + '_output.png'))

    plt.show()

    return


def test():
    reference_image = iio.imread("/home/edify/Code/Machine_Leaning/template.jpg")  # 参考模板图片
    img_r = reference_image[:, :, 0]
    img_g = reference_image[:, :, 1]
    img_b = reference_image[:, :, 2]
    reference_image_gray = img_r * 0.299 + img_g * 0.587 + img_b * 0.114
    plt.imshow(reference_image)
    detect_s = general_hough_closure(reference_image_gray)
    test_general_hough(detect_s, reference_image_gray, "/home/edify/Code/Machine_Leaning/query.jpg")


if __name__ == '__main__':
    test()