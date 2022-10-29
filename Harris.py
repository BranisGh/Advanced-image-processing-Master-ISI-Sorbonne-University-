from scipy import ndimage as ndi
import numpy as np
from scipy import signal as sig
from skimage.color import rgb2gray


def gradients_x_y(img_gray: np.ndarray) -> tuple:
    """
    Renvoie les dérivée de l'image en niveaux de gris suivant l'axe des x et y

    @ paramètres
    ------------
        img_gray : image en niveaux de gris.
    
    @ return
    ------------
        gradient_x, gradient_y : image en niveaux de gris (dérivée de 'img_gray' suivant l'axe des x et y).

    """
    assert len(img_gray.shape) == 2

    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]  )
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_x = sig.convolve2d(img_gray, kernel_x, mode='same')
    gradient_y = sig.convolve2d(img_gray, kernel_y, mode='same')

    return (gradient_x, gradient_y)

  

def non_maximum_suppression(img_corners: np.ndarray,
                            k_size: tuple=(3, 3)) -> np.ndarray:
    """
   L'algorithme consiste à parcourir tous les points d'intérêts, 
   et vérifier si ce dernier possède la plus grande intenté pour parmi les pixels adjacent le gardé, 
   si non le mètre à zéro  
    @ paramètres
    ------------
        img_corners : images en niveau de gris
        k_size      : les dimensions de la fenêtre entourant les pixels adjacents au point d'intérêt en question  
        
    @ return
    ------------
        img_corners : images en noire et blanc contenant tous les points d'internes en blanc 
    """

    assert k_size[0]%2 != 0 and k_size[1]%2 != 0
    dx, dy = k_size[0]//2, k_size[1]//2 
    rows, cols = img_corners.shape
    img_corners_ = np.zeros((rows+2*dx, cols+2*dy))
    rows, cols = img_corners_.shape
    img_corners_[dx:rows-dx, dy:cols-dy] = img_corners
    corners = np.argwhere(img_corners_ > 0)

    for (row, col) in corners:
        
        window = img_corners_[row-dx:row+dx+1, col-dy:col+dy+1]
        if np.sum(window)==0:
            localMax=0
        else:
            localMax = np.amax(window)
        maxCoord=np.unravel_index(np.argmax(window), window.shape) + np.array((row,col))
        # suppress everything
        img_corners_[row-dx:row+dx+1, col-dy:col+dy+1]=0
        # reset only the max
        if localMax > 0:
            img_corners_[tuple(maxCoord)[0]-dx, tuple(maxCoord)[1]-dy] = localMax

    img_corners = img_corners_[dx:rows-dx, dy:cols-dy]
    return img_corners



def harris( img_gray: np.ndarray, 
            k_size: tuple=(3, 3),
            filter: str='Gaussian',
            sd: int=5/6,
            truncate: int=4.0,
            supression: bool=True,
            k: float=0.05,
            harris_threshold: float=0.02   ) -> np.ndarray:
    """
    img_gray   : Image d'entrée, elle doit être en niveaux de gris et de type float32.
    k_size     : dimensions de la fenêtre de la suppression non maximal.
    sd         : Écart type du filtre Gaussien.
    supression : Boolean indiquant si on doit appliquer la suppression non maximal.    
    k          : Paramètre libre du détecteur de Harris dans l'équation, comprit entre {0.04, 0.05, 0.06}.
    """
    assert k in [0.04, 0.05, 0.06]
    assert filter.title() in ['Gaussian'.title(), 'Rectangle'.title()]

    # Étape 1 : Convertir l'image en niveaux de gris.
    """Si l'image en entrées est de type RGB ou BGR
    On doit la convertir en niveau de gris."""
    if len(img_gray.shape) > 2:
        img_gray = rgb2gray(img_gray)

    # Étape 2 : Calcul de la dérivée spatiale suivant l'axe de x et de y.
    """ https://linuxtut.com/fr/3ab2a67c78eafd456c32/"""
    # Filtre de sobel suivant l'axe des x et y avec un noyeau de (ksize X ksize).
    Ix, Iy = gradients_x_y(img_gray) 

    # Étape 3 : configuration du tenseur de structure.
    """https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1"""

    if filter.title() == 'Gaussian'.title():
        # Filtre gaussien de taille (5x5) et d'écart-type sd.
        # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
        Ixx = ndi.gaussian_filter(Ix**2, sigma=sd, truncate=truncate)
        Ixy = ndi.gaussian_filter(Iy*Ix, sigma=sd, truncate=truncate)
        Iyy = ndi.gaussian_filter(Iy**2, sigma=sd, truncate=truncate)

    elif filter.title() == 'Rectangle'.title():
        # Filtre rectangulaire de taille (5x5) et d'écart-type sd.
        Ixx = sig.convolve2d(Ix**2, np.ones((5, 5)), mode='same')
        Ixy = sig.convolve2d(Iy*Ix, np.ones((5, 5)), mode='same')
        Iyy = sig.convolve2d(Iy**2, np.ones((5, 5)), mode='same')

    # Étape 4 : Calcul de la réponse de Harris.
    # determinant.
    det = Ixx * Iyy - Ixy**2
    # trace.
    trace = Ixx + Iyy
    # reponse de Harris.
    harris_response = det - k * trace**2
    # Supprimer tous les pixels qui ne sont pas des poits d'intérêt.
    harris_response[harris_response < harris_threshold] = 0
    harris_corners = harris_response

    # Appliquer la suppression non maximal.
    if supression:    
        harris_corners = non_maximum_suppression(harris_corners, k_size=k_size)
    
    # img_corners = cv.dilate(img_corners, None)
    new_corners = np.argwhere(harris_corners > 0)

    return (harris_corners, new_corners)
            
