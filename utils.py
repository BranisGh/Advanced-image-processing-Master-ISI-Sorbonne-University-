import numpy as np
import matplotlib.pyplot as plt
import cv2


# method to show images as grid
def show_images(images: np.ndarray, 
                targets: str, 
                nb_images: int=40, 
                grid: bool=False, 
                total_cols: int=4, 
                figsize=(30, 20)) -> None: 
    """  
    showing images as grid, 
    images : Image dataset
    target : Images labels
    nb_images : Number of images to display
    grid : Boulen indicating whether the figure will be in grid mode or not  
    total_cols : Desired number of columns
    figsize : The dimensions of the figure 
    """
    # assert len(images) > 0
    # assert isinstance(images[0], np.ndarray)
    # compute number of cols & row  
    total_cols    = min(nb_images, total_cols)
    total_rows    = int(nb_images / total_cols) + (1 if nb_images % total_cols != 0 else 0)
    # Create a grid of subplots.
    fig, axes = plt.subplots(total_rows, total_cols, figsize=figsize)
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        # https://stackoverflow.com/questions/46862861/what-does-axes-flat-in-matplotlib-do
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]
    # it will helps to show total images as grid 
    for label, i in zip(targets[:nb_images], range(nb_images)):
        img = images[i]
        list_axes[i].imshow(img, cmap='gray')
        list_axes[i].grid(grid)
        list_axes[i].set_axis_off()
        list_axes[i].set_title(f'{label}')

    for i in range(nb_images, len(list_axes)):
        list_axes[i].set_visible(False)
    

def rotate_img(image: np.ndarray, angle: int=0, scale: int=1) -> np.ndarray:
    """
    image           : Image d'entrée.
    angle : angle de rotation de l'image de l'entrée.
    """

    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    

    return rotated


def coloring_points_interest(images_cornes: list, images: list, color: tuple=(0, 255, 0)) -> list: 
    """
    images_cornes : Liste d'image contenat chaqu'une des points d'intérêt.
    images        : Liste d'images en RGB ou BGR.
    color         : Couleur des point d'intérêt.
    """
    results = []
    for ci, im in zip(images_cornes, images):
        imm = im.copy()
        imm[ci > 0.0] = color
        results.append(imm)
    return results




"""
assert k_size[0]%2 != 0 and k_size[1]%2 != 0
dx, dy = k_size
rows, cols = img_corners.shape

for x in range(0, rows-dx+1):
    for y in range(0,cols-dy+1):
        window = img_corners[x:x+dx, y:y+dy]
        if np.sum(window)==0:
            localMax=0
        else:
            localMax = np.amax(window)
        maxCoord=np.unravel_index(np.argmax(window), window.shape) + np.array((x,y))
        # suppress everything
        img_corners[x:x+dx, y:y+dy]=0
        # reset only the max
        if localMax > 0:
            img_corners[tuple(maxCoord)] = localMax
"""