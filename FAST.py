import numpy as np
from skimage.color import rgb2gray
import sys


def circle(row: int, col: int) -> dict:
    """
    Renvoie un dictionnaire de pixels ((x,y) tuples) qui constituent la circonférence de la région de recherche d'un pixel.
    Circonférence du cercle = 16 pixels.
    Voir : http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html pour plus de détails.

    @ paramètres
    ------------
        row : coordonnée ligne d'un pixel.
        col : coordonnée colonne d'un pixel.
    
    @ return
    ------------
        circle_points : vecteur comparants 16 pixels, constituen la circonférence du pixel (row, col).
    """
    circle_points = {}
    circle_points['p1' ]  = (row-3, col  )    ;    circle_points['p2' ]  = (row-3, col+1) 
    circle_points['p3' ]  = (row-2, col+2)    ;    circle_points['p4' ]  = (row-1, col+3)
    circle_points['p5' ]  = (row, col+3  )    ;    circle_points['p6' ]  = (row+1, col+3)
    circle_points['p7' ]  = (row+2, col+2)    ;    circle_points['p8' ]  = (row+3, col+1)
    circle_points['p9' ]  = (row+3, col  )    ;    circle_points['p10']  = (row+3, col-1)
    circle_points['p11']  = (row+2, col-2)    ;    circle_points['p12']  = (row+1, col-3)
    circle_points['p13']  = (row, col-3  )    ;    circle_points['p14']  = (row-1, col-3)
    circle_points['p15']  = (row-2, col-2)    ;    circle_points['p16']  = (row-3, col-1)
    
    return circle_points


def isCorner(img_gray: np.ndarray, circle_points: list, row: int, col: int, t: int=9, n: int=12) -> bool:
    """
    Nous utilisons une version du test haute vitesse (voir la référence OpenCV) pour détecter un coin :
    Utilise les mêmes pixels retournés par la fonction cercle. 
    Les pixels sont ordonnés selon la référence OpenCV (voir la section intitulée : Feature Detection using FAST).
   
    Méthode :
        - Pour rendre l'algorithme rapide, comparez d'abord l'intensité des pixels {1, 5, 9 et 13} du cercle avec Ip . 
          Au moins trois de ces quatre pixels doivent satisfaire le critère de seuil pour que le point d'intérêt existe.
        
        - Si au moins trois des valeurs des quatre pixels {I1, I5, I9, I13} ne sont pas supérieures à Ip + tou inférieures à Ip - t, 
          alors p n'est pas un point d'intérêt (coin). Dans ce cas, rejetez le pixel p comme point d'intérêt possible. 
          Sinon, si au moins trois des pixels sont au-dessus ou en dessous de Ip + t, Ip - t respectivement, 
          alors vérifiez les 16 pixels et vérifiez si 12 pixels contigus relèvent du critère.
        
        - Répétez la procédure pour tous les pixels de l'image.

    @ paramètres
    ------------
        img_gray : images en niveau de gris [0, 1]
        row      : coordonnée ligne d'un pixel.
        col      : coordonnée colonne d'un pixel.
        t, n     : Paramètres à fixer t est empiriquement fixé à 9 et n > 8  
        
    @ return
    ------------
        True ou False
    """

    # Comparez d'abord l'intensité des pixels {1, 5, 9 et 13} du cercle avec Ip
    dark_points  = [circle_points[0], circle_points[4], circle_points[8], circle_points[12]] >= img_gray[row, col] + t 
    clear_points = [circle_points[0], circle_points[4], circle_points[8], circle_points[12]] <= img_gray[row, col] - t

    if sum(dark_points)>=3 or sum(clear_points)>=3:
        dark_points = np.array(circle_points) >= img_gray[row, col] + t
        clear_points = np.array(circle_points) <= img_gray[row, col] - t
        ones  = np.convolve(dark_points.astype(int).flatten() , np.ones(n),'valid')
        zeros = np.convolve(clear_points.astype(int).flatten() , np.ones(n),'valid')
        return ones.max()>=n or zeros.max()>=n
    return False


def calculateScore(img_gray: np.ndarray, circle_points: list, point: tuple) -> float:
    """ 
    Calcule le score de suppression non-maximale. 
    Le score V est défini comme la somme de la différence absolue entre les intensités de tous les points renvoyés 
    par la fonction cercle et l'intensité du pixel central. 
    tous les points renvoyés par la fonction cercle et l'intensité du pixel central.

    @ paramètres
    ------------
        img_gray : image en niveau de gris
        point    : cordonnées ligne et colonne d'un pixel

    @ return
    ------------
        True ou False
    """
    row, col = point
    score = sum([np.abs(img_gray[row, col] - value) for value in circle_points])
    return score


def non_maximum_suppression(img_corners: np.ndarray, k_size: tuple=(3, 3)) -> np.ndarray:
    """
    Effectue une suppression non-maximale sur la liste des coins.
    Pour les coins adjacents, élimine celui qui a le plus petit score.
    Sinon, ne rien faire.
    Puisque nous itérons à travers tous les pixels de l'image dans l'ordre, tous les points d'angle 
    adjacents devraient se trouver l'un à côté de l'autre dans la liste de tous les coins.
    Une suppression non maximale rejette les coins adjacents qui sont le même point dans la vie réelle.

    @ paramètres
    ------------
        img_corners : images en niveau de gris [0, 1]
        strides     :
        padding     :
        
    @ return
    ------------
        corners_supress : liste des coordonnées après suppression non maxima
    """
    
    dx, dy = k_size[0]//2, k_size[1]//2
    corners = np.argwhere(img_corners>0)
    for (row, col) in corners:
        window = img_corners[row-dx:row+dx+1, col-dy:col+dy+1]
        if np.sum(window) == 0:
            localMax = 0 
        else:
            localMax = np.amax(window)
    
        maxCoord = np.unravel_index(np.argmax(window), window.shape) + np.array((row, col))
        # suppress everything
        img_corners[row-dx:row+dx+1, col-dy:col+dy+1] = 0
        # reset only the max
        if localMax > 0:
            img_corners[tuple(maxCoord)[0]-dx, tuple(maxCoord)[1]-dy] = localMax
        

    return img_corners





def fast(img_gray: np.ndarray, suppress: bool=True, k_size: tuple=(3, 3)) -> np.ndarray:
    """
    Méthode renvoie si le pixel à la position (row, col) est un point d'intérêt ou pas
    @ paramètres
    ------------
        img_gray : images en niveau de gris [0, 1]
        padding  : taille de rembourrage

    @ return
    ------------
        img_corners : images en noire et blanc décrivant les points d'intérêts
    """
    assert k_size[0]%2 != 0 and k_size[1]%2 != 0

    """Si l'image en entrées est de type RGB ou BGR
    On doit la convertir en niveau de gris."""
    if len(img_gray.shape) > 2:
        img_gray = rgb2gray(img_gray)*255.0

    dx, dy = k_size
    img_corners = np.zeros(img_gray.shape)
    rows, cols = img_gray.shape
    
    for i in range(dx-1, rows-dy):
        for j in range(dx-1, cols-dy):
            circle_points = [img_gray[value] for _, value in circle(i, j).items()]
            if isCorner(img_gray, circle_points, row=i, col=j):
                img_corners[i, j] = calculateScore(img_gray, circle_points, (i, j))

    if suppress:
        img_corners = non_maximum_suppression(img_corners, k_size)
    
    new_corners = np.argwhere(img_corners>0)
    return img_corners, new_corners


