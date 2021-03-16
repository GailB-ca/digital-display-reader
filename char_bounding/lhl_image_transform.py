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


def img_process(img_array, gray=True, 
                flatten=True, 
                close_erode=True, 
                flatten_2=True,
                threshold_block_size=91,
                erode_iterations=2
               ):
    
    # Threshold block size should be lower values for noisey images
    # Erode iterations should be higher for noisey images
    
    image = img_array.copy()
    
    if(gray):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    if(flatten):
        image = cv2.adaptiveThreshold(image.copy(), 
                                      255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 
                                      threshold_block_size, 
                                      2)
        #image = cv2.threshold(image.copy(),55, 255, cv2.THRESH_BINARY_INV)[1]
        #image = 255 - image

    # denoise = cv2.fastNlMeansDenoising(thresh,None,7,7,21)
    
    if(close_erode):
        kernel = np.ones((10,10),np.uint8)
        image = cv2.morphologyEx(image.copy(), cv2.MORPH_CLOSE, kernel) 
        image = cv2.erode(image.copy(),kernel,
                          iterations = erode_iterations)

    if(flatten_2):
        image = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_BINARY)[1]

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
    
    
def extract_bounded_image(image_path, x_min, y_min, width, height, is_portrait=True):
    x = x_min
    y = y_min
    # print("{0} co-ords x {1} - {2}".format(image_path, x, y))
    w = width
    h = height
    image = cv2.imread(image_path)

    if (not is_portrait):
        # flip the image
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 

    num_img = image[y:y+h, x:x+w]

    if (not is_portrait):
        # flip back
        num_img = cv2.rotate(num_img, cv2.ROTATE_90_CLOCKWISE) 
        
    return num_img
