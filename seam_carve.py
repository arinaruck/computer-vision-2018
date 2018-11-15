from math import floor
from os import environ
from os.path import join
from pickle import load
from sys import argv
import numpy as np
import cv2
from skimage.io import imread, imsave, imshow


def seam_carve(img, mode, mask):
    img_out = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    y, x = np.gradient(img_out, 0.5)
    h, w = img_out.shape
    x[:, 0] = x[:, 0] / 2
    x[:, w - 1] = x[:, w - 1] / 2
    y[0, :] = y[0, :] / 2
    y[h - 1, :] = y[h - 1, :] / 2
    x = np.power(x, 2)
    y = np.power(y, 2)
    energy = x + y
    energy = np.power(energy, 0.5)
    h, w = img_out.shape
    orientation, direction = mode.split(' ')
    if mask is not None:
        mask = mask.astype('int')
        mask *= 256 * h * w
        energy += mask
    if direction == 'shrink':
        return shrink(orientation, img_out, img, energy, w, h, mask)
    return expand(orientation, img_out, img, energy, w, h, mask)


def get_seam_energy(orientation, energy, w, h):
    if orientation == 'horizontal':
        seam_energy = np.ones((h, w + 2))
        seam_energy[1 : -1, 1 : -1] = 0
        seam_energy *= 256 * 256 * h * w
        seam_energy[:, 1 : -1] = energy.copy()
        for i in range(1, h):
            for j in range(1, w + 1):
                seam_energy[i][j] = min([seam_energy[i - 1][j - 1], seam_energy[i - 1][j], seam_energy[i - 1][j + 1]]) \
                                    + energy[i][j - 1]  #  energy[i][j - 1] j - 1 because in energy j changes from 0 to w - 1 and in seam_energy it's shifted
    else:
        seam_energy = np.ones((h + 2, w))
        seam_energy[1: -1, 1: -1] = 0
        seam_energy *= 256 * 256  * h * w
        seam_energy[1 : -1, :] = energy.copy()
        for j in range(1, w):
            for i in range(1, h + 1):
                seam_energy[i][j] = min([seam_energy[i - 1][j - 1], seam_energy[i][j - 1], seam_energy[i + 1][j - 1]]) \
                                    + energy[i - 1][j]  #  energy[i - 1][j] i - 1 because in energy i changes from 0 to h - 1 and in seam_energy it's shifted
    return seam_energy


def get_next_seam(orientation, img, seam_energy, energy, last_pos):
    seam = [last_pos]
    x, y = last_pos
    if orientation == 'horizontal':
        img[y][x - 1] = -1
        # energy[y][x - 1] += h * w   needed if planning to do it multiple times, for that need 2 additional args w, h
        while y > 0:
            if seam_energy[y][x] == seam_energy[y - 1][x - 1] + energy[y][x - 1]:
                x -= 1
            elif (seam_energy[y][x] == seam_energy[y - 1][x + 1] + energy[y][x - 1]) and \
                    (seam_energy[y - 1][x + 1] != seam_energy[y - 1][x]):
                x += 1
            y -= 1
            seam.append((x, y))
            # energy[y][x - 1] += h * w
            img[y][x - 1] = -1
    else:
        img[y - 1][x] = -1
        # energy[y - 1][x] += h * w
        while x > 0:
            if seam_energy[y][x] == seam_energy[y - 1][x - 1] + energy[y - 1][x]:
                y -= 1
            elif (seam_energy[y][x] == seam_energy[y + 1][x - 1] + energy[y - 1][x]) and \
                    (seam_energy[y + 1][x - 1] != seam_energy[y][x - 1]):
                y += 1
            x -= 1
            seam.append((x, y))
            #energy[y - 1][x] += h * w
            img[y - 1][x] = -1


def shrink(orientation, img, original, energy, w, h, mask=None):
    seam_mask = np.zeros((h, w), dtype=int)
    seam_energy = get_seam_energy(orientation, energy, w, h)
    if orientation == 'horizontal':
        next_seam = np.argmin(seam_energy[-1, :])
        get_next_seam(orientation, img, seam_energy, energy, (next_seam, h - 1))
        shrinked = np.ndarray(shape=(h, w - 1, 3))
        for i in range(h):
            new_j = 0
            for j in range(w):
                if img[i][j] == -1:
                    seam_mask[i][j] = 1
                else:
                    shrinked[i][new_j] = original[i][j]
                    new_j += 1
    else:
        next_seam = np.argmin(seam_energy[:, -1])
        get_next_seam(orientation, img, seam_energy, energy, (w - 1, next_seam))
        shrinked = np.ndarray(shape=(h - 1, w, 3))
        for j in range(w):
            new_i = 0
            for i in range(h):
                if img[i][j] == -1:
                    seam_mask[i][j] = 1
                else:
                    shrinked[new_i][j] = original[i][j]
                    new_i += 1
    if mask is not None:
        mask += 2 * seam_mask
    return shrinked, mask, seam_mask


def expand(orientation, img, original, energy, w, h, mask=None):
    seam_mask = np.zeros((h, w), dtype=int)
    seam_energy = get_seam_energy(orientation, energy, w, h)
    if orientation == 'horizontal':
        next_seam = np.argmin(seam_energy[-1, :])
        get_next_seam(orientation, img, seam_energy, energy, (next_seam, h - 1))
        expanded = np.ndarray(shape=(h, w + 1, 3))
        for i in range(h):
            new_j = 0
            for j in range(w - 1):
                if img[i][j] == -1:
                    seam_mask[i][j] = 1
                    expanded[i][new_j] = original[i][j]
                    new_j += 1
                    expanded[i][new_j][0], expanded[i][new_j][1], expanded[i][new_j][2] = \
                        original[i][j][0] / 2 + original[i][j + 1][0] / 2,\
                        original[i][j][1] / 2 + original[i][j + 1][1] / 2,\
                        original[i][j][2] / 2 + original[i][j + 1][2] / 2

                else:
                    expanded[i][new_j] = original[i][j]
                new_j += 1
            j = w - 1
            if img[i][j] == -1:
                seam_mask[i][j] = 1
                expanded[i][new_j] = original[i][j]
                new_j += 1
                expanded[i][new_j][0], expanded[i][new_j][1], expanded[i][new_j][2] = \
                    original[i][j][0] / 2 + original[i][j - 1][0] / 2, \
                    original[i][j][1] / 2 + original[i][j - 1][1] / 2, \
                    original[i][j][2] / 2 + original[i][j - 1][2] / 2
            else:
                expanded[i][new_j] = original[i][j]
            new_j += 1
        expanded[:, -1] = original[:, -1]
    else:
        next_seam = np.argmin(seam_energy[:, -1])
        get_next_seam(orientation, img, seam_energy, energy, (w - 1, next_seam))
        expanded = np.ndarray(shape=(h + 1, w, 3))
        for j in range(w):
            new_i = 0
            for i in range(h - 1):
                if img[i][j] == -1:
                    seam_mask[i][j] = 1
                    expanded[new_i][j] = original[i][j]
                    new_i += 1
                    expanded[new_i][j][0], expanded[new_i][j][1], expanded[new_i][j][2] = \
                        original[i][j][0] / 2 + original[i + 1][j][0] / 2, \
                        original[i][j][1] / 2 + original[i + 1][j][1] / 2, \
                        original[i][j][2] / 2 + original[i + 1][j][2] / 2

                else:
                    expanded[new_i][j] = original[i][j]
                new_i += 1
            i = h - 1
            if img[i][j] == -1:
                seam_mask[i][j] = 1
                expanded[new_i][j] = original[i][j]
                new_i += 1
                expanded[new_i][j][0], expanded[new_i][j][1], expanded[new_i][j][2] = \
                    original[i][j][0] / 2 + original[i - 1][j][0] / 2, \
                    original[i][j][1] / 2 + original[i - 1][j][1] / 2, \
                    original[i][j][2] / 2 + original[i - 1][j][2] / 2

            else:
                expanded[new_i][j] = original[i][j]
            new_i += 1
        expanded[-1, :] = original[-1, :]
    if mask is not None:
        mask += 2 * seam_mask
    return expanded, mask, seam_mask


def check_test(output_dir, gt_dir):
    correct = 0
    with open(join(output_dir, 'output_seams'), 'rb') as fout, \
         open(join(gt_dir, 'seams'), 'rb') as fgt:
        for i in range(8):
            mine = load(fout)
            gt = load(fgt)
            if mine == gt:
                correct += 1
            else:
                counter = 0
                for my_el, gt_el in zip(reversed(mine), reversed(gt)):
                    if my_el != gt_el:
                        counter += 1
                print(counter)
    print(correct)
    return 'Ok %d/8' % correct


def grade(results_list):
    ok_count = 0
    for result in results_list:
        r = result['result']
        if r.startswith('Ok'):
            ok_count += int(r[3:4])
    total_count = len(results_list) * 8
    mark = floor(ok_count / total_count / 0.1)
    description = '%02d / %02d' % (ok_count, total_count)
    return description, mark


def run_single_test(data_dir, output_dir):
    from numpy import where
    from os.path import join
    from pickle import dump
    from seam_carve import seam_carve
    from skimage.io import imread

    def get_seam_coords(seam_mask):
        coords = where(seam_mask)
        t = [i for i in zip(coords[0], coords[1])]
        t.sort(key=lambda i: i[0])
        return tuple(t)

    def convert_img_to_mask(img):
        return ((img[:, :, 0] != 0) * -1 + (img[:, :, 1] != 0)).astype('int8')

    img = imread(join(data_dir, 'img.png'))
    mask = convert_img_to_mask(imread(join(data_dir, 'mask.png')))
    #print(mask)
    with open(join(output_dir, 'output_seams'), 'wb') as fhandle:
        for m in (None, mask):
            for direction in ('shrink', 'expand'):
                for orientation in ('horizontal', 'vertical'):
                    seam = seam_carve(img, orientation + ' ' + direction,
                                      mask=m)[2]
                    dump(get_seam_coords(seam), fhandle)

run_single_test(argv[1], argv[1])
check_test(argv[1], argv[2])
