
import cv2
import math
from datetime import datetime

""""a few useful functions to make life easier
feel free to add functions as you need"""

def load(dir_name):
    """loads the specific image from 'images' folder
    returns it's matrix with 3 channels"""

    dir =  dir_name
                #correction cv2 problem 

    return cv2.cvtColor(cv2.imread(dir, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def save(image, format = "jpg"):
    """saves image into the 'save' folder"""

    cv2.imwrite("save/" + datetime.now().strftime("%m%d-%H%M%S") + "." + format, image)

def display_image(image):
    """A function to display image"""

    cv2.namedWindow("test")
    cv2.imshow("test", image)
    k = cv2.waitKey(0) & 0xFF