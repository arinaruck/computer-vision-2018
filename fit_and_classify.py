from os import environ
from os.path import join
from sys import argv
import cv2
import numpy as np
import itertools


def get_gradient(img, ksize=1):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize)
    rgb_mag, rgb_angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    h, w = img.shape[0], img.shape[1]
    g_mag = np.zeros(shape=(h, w))
    g_angle = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            g_mag[i][j] = max(rgb_mag[i, j, 0], rgb_mag[i, j, 1], rgb_mag[i, j, 2])
            g_angle[i][j] = rgb_angle[i][j][np.argmax([rgb_mag[i, j, 0], rgb_mag[i, j, 1], rgb_mag[i, j, 2]])]
    #g_angle = np.mod(g_angle, 180) #from 0 - 2 * Pi to 0 - Pi
    return g_mag, g_angle


def get_cell_HoGs(img_size, cell_size, bin_size, bin_count, g_magnitude, g_angle):
    cell_count = img_size // cell_size
    HoG = np.zeros(shape=(cell_count, cell_count, bin_count), dtype=float)
    for i in range(cell_count):
        for j in range(cell_count):
            for k in range(cell_size):
                for l in range(cell_size):
                    angle = g_angle[i * cell_size + k][j * cell_size + l]
                    first_bin = int(angle // bin_size) % bin_count
                    next_bin = (first_bin + 1) % bin_count
                    share = angle % bin_size / bin_size
                    HoG[i][j][first_bin] += g_magnitude[i * cell_size + k][j * cell_size + l] * (1 - share)
                    HoG[i][j][next_bin] += g_magnitude[i * cell_size + k][j * cell_size + l] * share
    return HoG


def normalize(cells, bin_count):
    norm = 0.000005
    cells = list(itertools.chain.from_iterable(cells))
    cells_sq = np.power(cells, 2)
    norm += sum(cells_sq) ** 0.5
    normalized = np.divide(cells, norm)
    return normalized


def get_block_HoGs(img_size, cell_size, bin_count, cell_HoG):
    block_count = img_size // cell_size - 1
    HoG = []
    for i in range(block_count):
        for j in range(block_count):
            cells = [cell_HoG[i][j], cell_HoG[i][j + 1], cell_HoG[i + 1][j], cell_HoG[i + 1][j + 1]]
            HoG.append(normalize(cells, bin_count))
    return list(itertools.chain.from_iterable(HoG))


def extract_hog(img):
    img_size = 48
    img = cv2.resize(img, (img_size, img_size))
    img = np.float32(img) / 255.0
    k_size = 1  #  can be either 1 or 3
    g_magnitude, g_angle = get_gradient(img)
    cell_size = 6
    block_size = 2 * cell_size
    bin_count = 18
    bin_size = 20
    cell_HoG = get_cell_HoGs(img_size, cell_size, bin_size, bin_count, g_magnitude, g_angle)
    return get_block_HoGs(img_size, cell_size, bin_count, cell_HoG)


def fit_and_classify(train_features, train_labels, test_features):
    from sklearn import svm
    clf = svm.SVC(kernel='linear', C=0.5)
    clf.fit(train_features, train_labels)
    test_lables = clf.predict(test_features)
    return test_lables
