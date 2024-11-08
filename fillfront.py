import sys
import cv2
import numpy as np
from numba import jit
import warnings
import time
import math
warnings.filterwarnings("ignore")
# 拉普拉斯算子 锐化卷积核计算的是中心像素减去周围像素的差值（中心权重为正，周边权重为负）；
# 而Laplace算子则是周围像素之和减去中心像素的差值（中心权重为负，周边权重为正）
Lap = np.array([[ 1.,  1.,  1.],[ 1., -8.,  1.],[ 1.,  1.,  1.]])
kerx = np.array([[ 0.,  0.,  0.], [-1.,  0.,  1.], [ 0.,  0.,  0.]])
kery = np.array([[ 0., -1.,  0.], [ 0.,  0.,  0.], [ 0.,  1.,  0.]])

def IdentifyTheFillFront(masque, source):
    """ Identifie le front de remplissage """
    dOmega = []
    normale = []

    # 拉普拉斯滤波算遮盖部分边缘 后两个分别算遮盖部分的水平竖直方向边缘
    lap = cv2.filter2D(masque, cv2.CV_32F, Lap)
    GradientX = cv2.filter2D(masque, cv2.CV_32F, kerx)
    GradientY = cv2.filter2D(masque, cv2.CV_32F, kery)


    xsize, ysize = lap.shape

    x_coords, y_coords = np.where(lap > 0)
    dOmega = list(zip(y_coords, x_coords))
    dx = GradientX[x_coords, y_coords]
    dy = GradientY[x_coords, y_coords]
    N = np.sqrt(dy ** 2 + dx ** 2)
    normale = np.column_stack((dy / N, -dx / N))
    normale[N == 0] = np.column_stack((dy[N == 0], -dx[N == 0]))


    return(dOmega, normale)




