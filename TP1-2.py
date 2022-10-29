import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import os 


# Importer des images et les stockées dans une liste. 
current_directory = os.path.dirname(os.path.abspath('__file__'))
full_path_images = os.path.join(current_directory, 'Docs_Primitives', 'pics')
images = [cv.imread(os.path.join(full_path_images, img)) 
          for img in os.listdir(full_path_images) if img.endswith(('JPG', 'PNG'))]
targets = [img for img in os.listdir(full_path_images) if img.endswith(('JPG', 'PNG'))]

image1 =  images[3]
imggray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

image2 = images[4]
imggray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)


def cv2_fast(image):
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create(threshold = 35)
    # find and draw the keypoints
    kp = fast.detect(image, None)
    corners = cv.KeyPoint_convert(kp).astype(int)
    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    return corners 

def affichage(image, corners):
    temp_image = np.copy(image)
    temp_image[corners] = [0,255,0]
    plt.imshow(cv.cvtColor(temp_image, cv.COLOR_BGR2RGB))
    plt.show()


def desctipteur_simple(image, imggray, corners, affiche = False):
    dX, dY = (9,9)
    temp_image = np.copy(image)
    windows = []

    for i, point in enumerate(corners):
            window = imggray[point[0]-dX:point[0]+dX, point[1]-dY:point[1]+dY].flatten()
            windows.append(window)
            if affiche :#and i%5 == 0:
                temp_image = cv.rectangle(temp_image, (point[0]-dX, point[1]-dY), (point[0]+dX, point[1]+dY), (255, 0, 0))
    print("Nb de windows :", i)
    if affiche:
        plt.imshow(cv.cvtColor(temp_image, cv.COLOR_BGR2RGB))
        plt.show()
    return windows
        

def matching(windows1, windows2, corners1, corners2, thershold=0.099):
    
    for wind1, corn1 in zip(windows1, corners1):
        for wind2, corn2 in zip(windows2, corners2):
            dist = np.linalg.norm(wind1, wind2)



#Match.affichage()
corners = cv2_fast(image1)
desctipteur_simple(image1, imggray1, corners, affiche = True)