# 参考https://blog.csdn.net/CYM_CR/article/details/112347712
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:07:36 2021

@author: Chen
"""

import cv2 as cv
import numpy as np

img = cv.imread('scenary.jpeg', cv.IMREAD_UNCHANGED)

#print(img)

pi = []
std = []
std3 = []
con = []
ent = []
hist, bins = np.histogram(img.flatten(), 256, [0, 256])    # 计算灰度直方图
cdf = hist.cumsum()  # 求当前元素之前的所有元素累积和
cdf_normalized = hist / cdf.max()   # 归一化
# 计算均值
for i in range(256):
    p = cdf_normalized[i]*i
    pi.append(p)
mean = sum((pi))
# 计算方差（二阶矩）
for i in range(256):
    s = (i-mean)**2*cdf_normalized[i]
    std.append(s)
standard = sum(std)
#计算三阶矩
for i in range(256):
    s3 = (i-mean)**3*cdf_normalized[i]
    std3.append(s3)
standard3 = sum(std3)
#计算一致性
for i in range(256):
    c = cdf_normalized[i]**2
    con.append(c)
consistency = sum(con)
#计算熵
for i in range(256):
    e = -(cdf_normalized[i]*np.log2(cdf_normalized[i]+0.00001))
    ent.append(e)
entropy = sum(ent)
print(mean)
print(standard)
print(standard3)
print(consistency)
print(entropy)
#print(cdf.max())
#print(np.apply_over_axes(np.sum, img, [0,1])/cdf.max())
#print(sum(cdf_normalized))
