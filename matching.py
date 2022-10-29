import numpy as np
import cv2
from skimage.color import rgb2gray
from scipy.spatial.distance import pdist


def descriptors(img_gray: np.ndarray, corners: np.ndarray, k_size: tuple=(9, 9)) -> list:
    if len(img_gray) > 2:
        img_gray = rgb2gray(img_gray)

    dx, dy = k_size[0]//2, k_size[1]//2
    windows = {}

    for (row, col) in corners:
        window = img_gray[row-dx-1:row+dx, col-dy-1:col+dy].flatten()
        if window.size and len(window)==(k_size[0]**2):
            windows[(row, col)] = window
    
    return windows

def insert_windows_in_the_image(image: np.ndarray, 
                                windows: dict, 
                                k_size: tuple=(3, 3), 
                                color: tuple=((0, 255, 0))) -> np.ndarray:
    temp_image = image.copy()
    dx, dy = k_size
    for (row, col), _ in windows.items():
        temp_image = cv2.rectangle(temp_image, (col-dx-1, row-dy-1), (col+dx, row+dy), color)
    return temp_image


def matching(windows1, windows2, threshold=0.5, metric='euclidean'):
    new_window1 = {}
    new_window2 = {}

    for (row1, col1), window1 in windows1.items():
        pos_dist = {}
        for (row2, col2), window2 in windows2.items():
            pos_dist[(row2, col2)] = pdist((window1, window2), metric=metric) # np.linalg.norm(window1 - window2)
    
        pos_dist = {key: val for key, val in sorted(pos_dist.items(), key = lambda ele: ele[1], reverse = False)}

        if list(pos_dist.values())[0] / list(pos_dist.values())[1] < threshold:
            if (row1, col1) not in new_window1:
                new_window1[(row1, col1)] = window1
            if list(pos_dist.keys())[0] not in new_window2:
                new_window2[list(pos_dist.keys())[0]] = list(pos_dist.values())[0]
    
    return new_window1, new_window2
        