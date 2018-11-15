#!/usr/bin/python3

from os import environ
from os.path import join
from sys import argv, exit
import numpy as np

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
    return coord_x_g - x_3


def get_y_b(coord_y_g, y_3, v_frame, pic_height):
    return coord_y_g + pic_height + 2 * v_frame - y_3


def get_x_r(coord_x_g, x_1):
    return coord_x_g + x_1


def get_y_r(coord_y_g, y_1, v_frame, pic_height) :
    return coord_y_g - pic_height - 2 * v_frame + y_1


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
            elif abs(x_3) < x_1:
                i_1 = i_1[:, x_1:]
                i_2 = i_2[:, : -x_1]
                i_3 = i_3[:, -x_3: x_1 + x_3]
            else:
                i_1 = i_1[:, x_1:]
                i_2 = i_2[:, :-x_1]
                i_3 = i_3[:, x_1:]

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
            elif x_3 < abs(x_1):
                i_1 = i_1[:, : x_1]
                i_2 = i_2[:, -x_1:]
                i_3 = i_3[:, -x_1 - x_3: -x_3]
            else:
                i_1 = i_1[:, : -x_3]
                i_2 = i_2[:, x_3:]
                i_3 = i_3[:, : -x_3]

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
            elif abs(y_3) < y_1:
                i_1 = i_1[y_1:, :]
                i_2 = i_2[: -y_1, :]
                i_3 = i_3[-y_3: y_1 + y_3, :]
            else:
                i_1 = i_1[y_1:, :]
                i_2 = i_2[: -y_1, :]
                i_3 = i_3[y_1:, :]

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
            elif y_3 < abs(y_1):
                i_1 = i_1[: y_1, :]
                i_2 = i_2[-y_1:, :]
                i_3 = i_3[-y_1 - y_3: -y_3, :]
            else:
                i_1 = i_1[: -y_3, :]
                i_2 = i_2[y_3:, :]
                i_3 = i_3[: -y_3, :]

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
    while (G.shape[0] > 220 or G.shape[1] > 220):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp


def shift(i_1, i_2, level=0, x_s=0, y_s=0):
    min_x_shift = 0
    min_y_shift = 0
    min = i_1.shape[0] * i_1.shape[1]
    lim = 12 - 3 * level
    for x_shift in range(-lim, lim + 1):
        x_shift += 2 * x_s
        for y_shift in range(-lim, lim + 1):
            y_shift += 2 * y_s
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

            curr = skimage_views_MSD_v1(i_1_cp, i_2_cp)[0][0]
            if curr < min:
                min = curr
                min_x_shift = x_shift
                min_y_shift = y_shift
    return min_x_shift, min_y_shift


def align(img, g_coord):
    from numpy.lib import scimath
    from skimage.io import imread, imsave

    coord_y_g, coord_x_g = (g_coord[0], g_coord[1])
    height, width = img.shape

    v_frame = int(height / 3 * 0.05)
    new_height = int((height - 6 * v_frame) / 3)  # height of the image without frame
    h_frame = int(width * 0.08)

    up = img[v_frame : v_frame + new_height + 1, h_frame : -h_frame];
    mid = img[3 * v_frame + new_height : 3 * v_frame + 2 * new_height + 1, h_frame : -h_frame];
    down = img[5 * v_frame + 2 * new_height : 5 * v_frame + 3 * new_height + 1, h_frame : -h_frame];

    p1 = Pyramid(up)
    p2 = Pyramid(mid)
    p3 = Pyramid(down)

    x_1, x_2, x_3, y_1, y_2, y_3 = 0, 0, 0, 0, 0, 0
    pyramids = len(p1)

    for i in range(pyramids - 1, -1, -1):
        lvl = pyramids - i - 1
        x_1, y_1 = shift(p1[i].copy(), p2[i].copy(), lvl, x_1, y_1)
        x_3, y_3 = shift(p2[i].copy(), p3[i].copy(), lvl, x_3, y_3)

    coord_x_r = get_x_r(coord_x_g, x_1)
    coord_y_r = get_y_r(coord_y_g, y_1, v_frame, new_height)
    coord_x_b = get_x_b(coord_x_g, x_3)
    coord_y_b = get_y_b(coord_y_g, y_3, v_frame, new_height)

    up, mid, down = slice(up, mid, down, x_1, x_3, y_1, y_3)

    res = cv2.merge((down, mid, up))

    return res, (coord_y_r, coord_x_r), (coord_y_b, coord_x_b)
