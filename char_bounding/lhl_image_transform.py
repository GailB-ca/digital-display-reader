import math
from typing import Tuple, Union
import cv2
import numpy as np
from deskew import determine_skew
from imutils.object_detection import non_max_suppression
import argparse
import time
from matplotlib import pyplot as plt
import pytesseract


def img_process(img_array, gray=True, flatten=True):
    image = img_array.copy()
    
    if(gray):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        
    if(flatten):
        image = cv2.threshold(image.copy(),55, 255, cv2.THRESH_BINARY_INV)[1]
        image = 255 - image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3)
    
    return image

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def img_rotate(img_array):
    image = img_array.copy()
    angle = determine_skew(image)
    return rotate(image, angle, (255, 255, 255))


def compute_bounding(img_array):
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
    boxes = pytesseract.image_to_boxes(img_array, 
                                  # config='--psm 11 --oem 3 -c tessedit_char_whitelist=123456789'
                                   config='--psm 11 --oem 3'
                                  )
    print(boxes)
    return boxes
    
def plot_boxes(img_array, boxes):    
    image = img_array.copy()
    (h, w) = image.shape[:2]
    for b in boxes.splitlines():
        #print(b)
        b = b.split(' ')
        image = cv2.rectangle(image, 
                        (int(b[1]), h - int(b[2])), 
                        (int(b[3]), h - int(b[4])), 
                        (0, 255, 255), 
                        20 # Pixel border
                       )
    
    plot_image(image)
   
    return image
    
    
def plot_image(image_array):
    plt.imshow(image_array, cmap="gray")
    plt.show()