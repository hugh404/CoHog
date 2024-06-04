import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

DTYPE_FLOAT = np.float64
DTYPE_INT = int
DTYPE_UINT8 = np.uint8

height = 0
width = 0
cell_size = 16
bin_size = 8
angle_unit = 360 / bin_size
img = np.zeros((height, width), dtype=DTYPE_UINT8)


def initialize(image, cellsize=16, binsize=8):
    global height
    global width
    global cell_size
    global bin_size
    global angle_unit
    global img
    img = image
    img = np.sqrt(img / float(np.max(img)))
    img = img * 255
    height, width = img.shape[0], img.shape[1]
    cell_size = cellsize
    bin_size = binsize
    angle_unit = 360 / binsize

    return height, width, angle_unit


def extract():
    global height
    global width
    global cell_size
    global bin_size
    global angle_unit
    global img

    itr = 0

    gradient_magnitude = np.zeros((height, width), dtype=DTYPE_FLOAT)
    gradient_angle = np.zeros((height, width), dtype=DTYPE_FLOAT)
    cell_magnitude = np.zeros((cell_size, cell_size), dtype=DTYPE_FLOAT)
    cell_angle = np.zeros((cell_size, cell_size), dtype=DTYPE_FLOAT)
    cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size), dtype=DTYPE_FLOAT)
    block_vector = []
    magnitude = 0.0

    hog_vector = np.zeros(((cell_gradient_vector.shape[0] - 1) * (cell_gradient_vector.shape[1] - 1), bin_size * 4),
                          dtype=DTYPE_FLOAT)

    gradient_magnitude, gradient_angle = global_gradient()
    gradient_magnitude = abs(gradient_magnitude)

    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_gradient_vector[i, j] = cell_gradient(cell_magnitude, cell_angle)

    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            magnitude = math.sqrt(sum(k ** 2 for k in block_vector))
            if magnitude != 0:
                block_vector = [element / magnitude for element in block_vector]
            hog_vector[itr] = block_vector
            itr += 1

    return hog_vector


def global_gradient():
    global height
    global width
    global img

    gradient_values_x = np.zeros((height, width), dtype=DTYPE_FLOAT)
    gradient_values_y = np.zeros((height, width), dtype=DTYPE_FLOAT)
    gradient_magnitude = np.zeros((height, width), dtype=DTYPE_FLOAT)
    gradient_angle = np.zeros((height, width), dtype=DTYPE_FLOAT)

    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    return gradient_magnitude, gradient_angle


def cell_gradient(cell_magnitude, cell_angle):
    global bin_size
    global angle_unit

    orientation_centers = np.zeros(bin_size, dtype=DTYPE_FLOAT)

    for i in range(cell_magnitude.shape[0]):
        for j in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle, max_angle, mod = get_closest_bins(gradient_angle)
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))

    return orientation_centers


def get_closest_bins(gradient_angle):
    global bin_size
    global angle_unit

    idx = int(gradient_angle / angle_unit)
    mod = gradient_angle % angle_unit
    if idx == bin_size:
        return idx - 1, idx % bin_size, mod
    return idx, (idx + 1) % bin_size, mod