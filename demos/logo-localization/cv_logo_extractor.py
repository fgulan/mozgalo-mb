import cv2
import imutils
import numpy as np
import sys
from matplotlib import pyplot as plt

def binarize_image_otsu(image):
    _, th_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th_image

def binarize_image_niblack(image, block_size=3, C=0):
    return cv2.ximgproc.niBlackThreshold(image, 255 , cv2.THRESH_BINARY, block_size, C)

def binarize_image_gauss(image, block_size=9, C=9):
    return cv2.adaptiveThreshold(image, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

def extract_external_contour(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    return max(contours, key=cv2.contourArea)

def contour_bounding_box(c):
    left = tuple(c[c[:, :, 0].argmin()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return left, top, right, bottom

def mask_image(image, contour):
    left, top, right, bottom = contour_bounding_box(contour)
    min_x, min_y = left[0], top[1]
    max_x, max_y = right[0], bottom[1]
    return image[min_y:max_y, min_x:max_x]

def get_line_edges(line):
    first = 0; last = len(line) - 1
    for index, value in enumerate(line):
        if value != 0:
            first = index; break
    for index, value in enumerate(reversed(line)):
        if value != 0:
            last = last - index; break
    return first, last

def stretch_color_line(line, width):
    orig_size = len(line)
    scale = orig_size / width
    new_line = np.ones((width, 3), np.uint8)
    last_index = orig_size - 1
    for i in range(width):
        new_line[i] = line[min(round(i * scale), last_index)]
    return new_line

def stretch_image(img_color, img_binary):
    height, width = img_binary.shape
    out_image = np.ones((height, width, 3), np.uint8)

    for i in range(height):
        first, last = get_line_edges(img_binary[i])
        line = img_color[i][first:last+1]
        new_line = stretch_color_line(line, width)
        for j in range(width):
            out_image[i][j] = new_line[j]
    return out_image

def smooth_histogram_iteration(xs):
    size = len(xs)
    new_xs = np.zeros(size)
    new_xs[0] = xs[0]
    new_xs[size - 1] = xs[size - 1]
    for i in range(1, size - 1):
        new_val = (new_xs[i - 1] + xs[i + 1]) / 2
        if new_val > xs[i]:
            new_xs[i] = new_val
        else:
            new_xs[i] = xs[i]
    return new_xs

def smooth_histogram(xs, iterations=1):
    for _ in range(iterations):
        xs = smooth_histogram_iteration(xs)
    return xs

def histogram_projection(bin_image, x_axis=False):
    size = bin_image.shape[0]
    if x_axis:
        size = bin_image.shape[1]
    values = np.zeros(size)
    for i in range(size):
        if x_axis:
            line = bin_image[:, i]
        else:
            line = bin_image[i]
        values[i] = np.sum(1 - (line / 255))
    return values

def longest_histogram_seq(values, mean):
    count = 0
    prev = 0
    indexend = 0
    for i in range(len(values)):
        if values[i] > mean:
            count += 1
        elif count > prev:
            prev = count
            indexend = i
        else: count = 0
    return indexend-prev, indexend-1

def extract_logo(input_image_path):
    image_color = cv2.imread(input_image_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    image_binarized = binarize_image_otsu(image_gray)

    receipt_contour = extract_external_contour(image_binarized)
    receipt_image_color = mask_image(image_color, receipt_contour)
    receipt_image_binarized = mask_image(image_binarized, receipt_contour)
    
    stretched_image_color = stretch_image(receipt_image_color, receipt_image_binarized)
    stretched_image_gray = cv2.cvtColor(stretched_image_color, cv2.COLOR_BGR2GRAY)
    stretched_image_binarized = binarize_image_gauss(stretched_image_gray)

    y_hist = histogram_projection(stretched_image_binarized)
    y_mean = np.mean(y_hist)
    y_hist = smooth_histogram(y_hist, iterations=5)
    start_y, end_y = longest_histogram_seq(y_hist, y_mean)


    x_hist = histogram_projection(stretched_image_binarized, x_axis=True)
    x_mean = np.mean(x_hist)
    x_hist = smooth_histogram(x_hist, iterations=5)
    start_x, end_x = longest_histogram_seq(x_hist, x_mean)

    return stretched_image_color[start_y:end_y, start_x:end_x]

if __name__ == '__main__':
    logo = extract_logo(sys.argv[1])
    cv2.imshow('logo', logo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()