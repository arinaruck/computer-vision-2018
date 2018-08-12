#!/usr/bin/python3

from os import environ
from os.path import join
from sys import argv, exit
import numpy as np
import cv2

from skimage.util import view_as_windows
from skimage.feature import match_template
import cv2
from cv2 import matchTemplate as cv2m
from scipy.ndimage.filters import uniform_filter as unif2d
from scipy.signal import convolve2d as conv2
from sklearn.metrics import mean_squared_error as mse


def skimage_views_MSD_v1(img, tmpl):
    return ((view_as_windows(img, tmpl.shape) - tmpl) ** 2).mean(axis=(2, 3))


def get_x_b(coord_x_g, x_3):
    #return coord_x_g + (x_1 - x_2 - x_3) // 2
    return coord_x_g  - x_3

def get_y_b(coord_y_g, y_3, v_frame, new_height):
    #return coord_y_g + v_frame + new_height + (y_1 - y_2 - y_3) // 2
    return coord_y_g + v_frame + new_height - y_3

def get_x_r(coord_x_g, x_1):
    #return coord_x_g + (x_1 + x_2 - x_3) // 2
    return coord_x_g + x_1

def get_y_r(coord_y_g, y_1, v_frame, new_height) :
    #return coord_y_g - v_frame - new_height + (y_1 + y_2 - y_3) // 2
    return coord_y_g - v_frame - new_height + y_1


def slice(i_1, i_2, i_3, x_1, x_3, y_1, y_3):
    if x_1 > 0:

        if x_3 > 0:
            i_1 = i_1[:, x_1 + x_3:]
            i_2 = i_2[:, x_3: -x_1]
            i_3 = i_3[:, : - x_3 - x_1]

        elif x_3 < 0:
            if abs(x_3) > x_1:
                i_1 = i_1[:, x_1: x_3 + x_1]
                i_2 = i_2[:, : x_3]
                i_3 = i_3[:, -x_3:]
            else:
                i_1 = i_1[:, x_1:]
                i_2 = i_2[:, : -x_1]
                i_3 = i_3[:, -x_3: x_1 + x_3]

        else:
            i_1 = i_1[:, x_1:]
            i_2 = i_2[:, : -x_1]
            i_3 = i_3[:, : -x_1]

    elif x_1 < 0:
        if x_3 > 0:
            if x_3 > abs(x_1):
                i_1 = i_1[:, x_1 + x_3: x_1]
                i_2 = i_2[:, x_3:]
                i_3 = i_3[:, : -x_3]
            else:
                i_1 = i_1[:, : x_1]
                i_2 = i_2[:, -x_1:]
                i_3 = i_3[:, -x_1 - x_3: -x_3]

        elif x_3 < 0:
            i_1 = i_1[:, :  x_1 + x_3]
            i_2 = i_2[:, -x_1: x_3]
            i_3 = i_3[:, -x_1 - x_3:]

        else:
            i_1 = i_1[:, :  x_1]
            i_2 = i_2[:, -x_1:]
            i_3 = i_3[:, -x_1:]

    else:
        if x_3 > 0:
            i_1 = i_1[:, x_3:]
            i_2 = i_2[:, x_3:]
            i_3 = i_3[:, : -x_3]

        elif x_3 < 0:
            i_1 = i_1[:, : x_3]
            i_2 = i_2[:, : x_3]
            i_3 = i_3[:, - x_3:]
    if y_1 > 0:

        if y_3 > 0:
            i_1 = i_1[y_1 + y_3:, :]
            i_2 = i_2[y_3: -y_1, :]
            i_3 = i_3[: -y_3 - y_1, :]

        if y_3 < 0:
            if abs(y_3) > y_1:
                i_1 = i_1[y_1: y_3 + y_1, :]
                i_2 = i_2[: y_3, :]
                i_3 = i_3[-y_3:, :]
            else:
                i_1 = i_1[y_1:, :]
                i_2 = i_2[: -y_1, :]
                i_3 = i_3[-y_3: y_1 + y_3, :]
        if y_3 == 0:
            i_1 = i_1[y_1:, :]
            i_2 = i_2[: -y_1, :]
            i_3 = i_3[: -y_1, :]

    elif y_1 < 0:

        if y_3 > 0:
            if y_3 > abs(y_1):
                i_1 = i_1[y_1 + y_3: y_1, :]
                i_2 = i_2[y_3:, :]
                i_3 = i_3[: -y_3, :]
            else:
                i_1 = i_1[: y_1, :]
                i_2 = i_2[-y_1:, :]
                i_3 = i_3[-y_1 - y_3: -y_3, :]

        if y_3 < 0:
            i_1 = i_1[: y_1 + y_3, :]
            i_2 = i_2[-y_1: y_3, :]
            i_3 = i_3[-y_1 - y_3:, :]

        if y_3 == 0:
            i_1 = i_1[: y_1, :]
            i_2 = i_2[-y_1:, :]
            i_3 = i_3[-y_1:, :]
    else:
        if y_3 > 0:
            i_1 = i_1[y_3:, :]
            i_2 = i_2[y_3:, :]
            i_3 = i_3[:, -y_3:]

        elif y_3 < 0:
            i_1 = i_1[: y_3, :]
            i_2 = i_2[: y_3, :]
            i_3 = i_3[-y_3:, :]
    return i_1, i_2, i_3

def Pyramid(i):
    # generate Gaussian pyramid
    G = i.copy()
    gp = [G]
    while (G.shape[0] > 200 or G.shape[1] > 200):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def MSE(i_1, i_2, min):
    #break into parts of fixed size, check whether sum is > min * h * w
    if i_1.shape != i_2.shape:
        print(i_1.shape, i_2.shape)
        return -1
    h, w = i_1.shape
    curr = min * h * w
    s = w // 5  #  shift
    st = 0
    end = s
    sum = 0
    while end < w:
        i_1_cp = i_1[st : end]
        i_2_cp = i_2[st : end]
        st += s
        end += s
        dif = np.subtract(i_1_cp, i_2_cp)
        dif = np.power(dif, 2)
        sum += np.sum(dif)
        if (sum >= curr) :
            return sum

    i_1_cp = i_1.copy()[end :]
    i_2_cp = i_2.copy()[end :]
    dif = np.subtract(i_1_cp, i_2_cp)
    dif = np.power(dif, 2)
    sum += np.sum(dif)

    res = 1.0 / (h * w) * sum
    return res


def cross_correlation(i_1, i_2):
    if i_1.shape != i_2.shape:
        print(i_1.shape, i_2.shape)
        return -1
    h, w = i_1.shape
    prod_sum = np.sum(np.multiply(i_1, i_2))
    i_1 = np.power(i_1, 2)
    i_2 = np.power(i_2, 2)
    sum_1 = np.sum(i_1)
    sum_2 = np.sum(i_2)
    res = 1.0 / (sum_1 * sum_2) ** 0.5 * prod_sum
    return res


def shift(i_1, i_2, level=0, x_s=0, y_s=0):
    min_x_shift = 0
    min_y_shift = 0
    max_x_shift = 0
    max_y_shift = 0
    min = i_1.shape[0] * i_1.shape[1]
    max = 0;
    lim = 20 - 2 * level
    #lim = 18
    shifts = [0]
    for i in range(1, lim + 1):
        shifts.append(i)
        shifts.append(-i)
    for x_shift in shifts:
        x_shift += x_s
        for y_shift in shifts:
            y_shift += y_s
            i_1_cp = i_1.copy()
            i_2_cp = i_2.copy()
            if x_shift > 0:
                i_1_cp = i_1_cp[: , x_shift : ]
                i_2_cp = i_2_cp[: , : -x_shift]
            elif x_shift < 0:
                i_1_cp = i_1_cp[: , : x_shift]
                i_2_cp = i_2_cp[: , -x_shift : ]
            if y_shift > 0:
                    i_1_cp = i_1_cp[y_shift : , :]
                    i_2_cp = i_2_cp[: -y_shift, :]
            elif y_shift < 0:
                    i_1_cp = i_1_cp[: y_shift , :]
                    i_2_cp = i_2_cp[-y_shift : , :]

            #curr = MSE(i_1_cp, i_2_cp, min)
            curr = skimage_views_MSD_v1(i_1_cp, i_2_cp)[0][0]
            if curr < min:
                min = curr
                min_x_shift = x_shift
                min_y_shift = y_shift
            '''
            curr = cross_correlation(i_1_cp, i_2_cp)
            if curr > max:
                max = curr
                max_x_shift = x_shift
                max_y_shift = y_shift
            if curr == -1:
                print(x_shift, y_shift)
             '''
    print(min_x_shift, min_y_shift, min)
    return min_x_shift, min_y_shift
    #return max_x_shift, max_y_shift



def align(img, g_coord) :
    from numpy.lib import scimath
    from skimage.io import imread, imsave
    import time

    start = time.time()

    coord_y_g, coord_x_g = (g_coord[0], g_coord[1])
    height, width = img.shape

    v_frame = int(height / 3 * 0.02)
    new_height = int((height - 6 * v_frame) / 3)  # height of the image without frame
    h_frame = int(width * 0.06)

    up = img[2 * v_frame : 2 * v_frame + new_height, h_frame : -h_frame];
    mid = img[3 * v_frame + new_height : 3 * v_frame + 2 * new_height, h_frame : -h_frame];
    down = img[4 * v_frame + 2 * new_height : height - 2 * v_frame, h_frame : -h_frame];

    h1, w1 = up.shape
    h2, w2 = mid.shape
    h3, w3 = down.shape
    h = min([h1, h2, h3])
    w = min([w1, w2, w3])
    up = up[: h, : w]
    mid = mid[: h, : w]
    down = down[: h, : w]

    end = time.time()
    print("Pic division", (end - start) / 60)

    start = time.time()

    p1 = Pyramid(up)
    p2 = Pyramid(mid)
    p3 = Pyramid(down)

    end = time.time()
    print("Pyramids", (end - start) / 60)

    start = time.time()
    x_1, x_2, x_3, y_1, y_2, y_3 = 0, 0, 0, 0, 0, 0
    pyramids = len(p1)

    print("levels: ", pyramids)
    for i in range(pyramids - 1, -1, -1):
        lvl = pyramids - i - 1
        x_1, y_1 = shift(p1[i].copy(), p2[i].copy(), lvl, x_1, y_1)
        x_3, y_3 = shift(p2[i].copy(), p3[i].copy(), lvl, x_3, y_3)

    coord_x_r = get_x_r(coord_x_g, x_1)
    coord_y_r = get_y_r(coord_y_g, y_1, v_frame, new_height)
    coord_x_b = get_x_b(coord_x_g, x_3)
    coord_y_b = get_y_b(coord_y_g, y_3, v_frame, new_height)


    end = time.time()
    print("Shifting", (end - start) / 60)
    #y_max = get_max(v_frame + new_height, v_frame + new_height - y_1, v_frame + new_height + y_2)
    #y_min = get_min(3 * v_frame + 2 * new_height, 3 * v_frame + 2 * new_height - y_1, 3 * v_frame + 2 * new_height + y_2)
    #x_max = get_max(0, -x_1, x_2)
    #x_min = get_min(w, w - x_1, w + x_2)
    #res = np.ndarray(shape=(y_min - y_max, x_min - x_max, 3), dtype=float)
    #print(y_max, y_min, x_max, x_min)
    '''
    for i in range(y_max, y_min):
        for j in range(x_max, x_min):
            res[i - y_max][j - x_max][0] = img[get_y_r(i, y_3, v_frame, new_height)][get_x_r(j, x_3)]
            res[i - y_max][j - x_max][1] = img[i][j]
            res[i - y_max][j - x_max][2] = img[get_y_b(i, y_1, v_frame, new_height)][get_x_b(j, x_1)]
    '''
    start = time.time()

    up, mid, down = slice(up, mid, down, x_1, x_3, y_1, y_3)


    end = time.time()
    print("Slicing", (end - start) / 60)

    start = time.time()

    res = cv2.merge((down, mid, up))

    end = time.time()
    print("Merging", (end - start) / 60)
    return res, (coord_y_r, coord_x_r), (coord_y_b, coord_x_b)


def run_single_test(data_dir, gt_dir):
    #from align import align
    from skimage.io import imread, imsave
    parts = open(join(data_dir, 'g_coord.csv')).read().rstrip('\n').split(',')
    g_coord = (int(parts[0]), int(parts[1]))
    img = imread(join(data_dir, 'img.png'), plugin='matplotlib')

    aligned_img, (b_row, b_col), (r_row, r_col) = align(img, g_coord)

    with open(join(gt_dir, 'gt.csv')) as fhandle:
        parts = fhandle.read().rstrip('\n').split(',')
        coords = map(int, parts[1:])
        gt_b_row, gt_b_col, _, _, gt_r_row, gt_r_col, diff_max = coords

    x_diff = abs(b_row - gt_b_row) + abs(r_row - gt_r_row)

    y_diff = abs(b_col - gt_b_col) + abs(r_col - gt_r_col)

    imsave("test.png", aligned_img)
    if x_diff + y_diff > diff_max:
        print(x_diff, y_diff)
        return 'Wrong answer'
    return 'Ok'


import math
from numpy.lib import scimath
from skimage.io import imread, imsave, imshow

import time
import sys

data_dir = sys.argv[1]
parts = open(join(data_dir, 'g_coord.csv')).read().rstrip('\n').split(',')
g_coord = (int(parts[0]), int(parts[1]))

img = imread(join(data_dir, 'img.png'), plugin='matplotlib')

start = time.time()

print(run_single_test(argv[1], argv[2]))

end = time.time()
print((end - start) / 60)
