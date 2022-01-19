import numpy as np
import matplotlib.pyplot as plt
import cv2


def swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b


def is_inside_line(point, line):
    xl1 = line[0][0]
    xl2 = line[1][0]
    if xl1 > xl2:
        xl1, xl2 = swap(xl1, xl2)
    yl1 = line[0][1]
    yl2 = line[1][1]
    if yl1 > yl2:
        yl1, yl2 = swap(yl1, yl2)
    return point[0] >= xl1 and point[0] <= xl2 and point[1] >= yl1 and point[1] <= yl2


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def sync_maps(icons_map_path, no_icons_map_path):
    im = cv2.imread(icons_map_path)
    nim = cv2.imread(no_icons_map_path)

    y1 = 106
    y2 = 928
    x1 = 950
    x2 = 1700

    crop_im = im[y1:y2, x1:x2]
    crop_nim = nim[y1:y2, x1:x2]
    cv2.imwrite("./images/maps/osrs_world_map_icons_cropped.png", crop_im)
    cv2.imwrite("./images/maps/osrs_world_map_no_icons_cropped.png", crop_nim)

    print(im.shape)
    print(nim.shape)
    print(crop_im.shape)
    print(crop_nim.shape)

    convert_to_bw('./images/maps/osrs_world_map_no_icons_cropped.png')


def convert_to_bw(image_path):
    image = cv2.imread(image_path)

    low_white = np.array([100, 100, 100])
    high_white = np.array([255, 255, 255])

    low_water = np.array([165, 118, 98])
    high_water = np.array([175, 130, 110])

    # create the Mask
    mask = cv2.inRange(image, low_white, high_white)
    mask = cv2.cvtColor(mask, cv2.IMREAD_GRAYSCALE)
    thresh = 127
    mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1]
    mask_water = cv2.inRange(image, low_water, high_water)
    mask_water = cv2.cvtColor(mask_water, cv2.IMREAD_GRAYSCALE)
    thresh = 127
    mask_water = cv2.threshold(mask_water, thresh, 255, cv2.THRESH_BINARY)[1]
    merge = np.maximum(mask, mask_water)
    # print(merge.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(merge, kernel, iterations=1)
    # cv2.imshow("merge", merge)
    # cv2.imshow("mask", mask)
    # cv2.imshow("water", mask_water)
    cv2.imshow("cam", dilate)
    cv2.waitKey(5000)
    cv2.imwrite("./images/maps/converted_bw_map_test.png", dilate)
    return dilate


def create_grid(image_path):
    image = cv2.imread(image_path, 0)
    #image = cv2.resize(image, (200, 200))
    image = cv2.bitwise_not(image)
    [np.where(line == 255, 1, line) for line in image]
    return image


def rescale_pos(pos, pos_scale, scale):
    return [int(pos[0] / pos_scale[0] * scale[0]), int(pos[1] / pos_scale[1] * scale[1])]


def create_path(route):
    path = []
    for i in range(len(route) - 1):
        path.append((route[i], route[i + 1]))
    return path


# extract x and y coordinates from route list
def plot_path(route, start, goal, grid):
    x_coords = []
    y_coords = []

    for i in (range(0, len(route))):
        x = route[i][1]
        y = route[i][0]
        x_coords.append(x)
        y_coords.append(y)
    # plot map and path
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap=plt.cm.Dark2)
    ax.scatter(start[0], start[1], marker="*", color="yellow", s=200)
    ax.scatter(goal[0], goal[1], marker="*", color="red", s=200)
    ax.plot(y_coords, x_coords, color="black")

    plt.show()
