import math
from typing import Tuple, Union
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
import time
from matplotlib import pyplot as plt


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


    
    
def plot_image(image_array):
    plt.imshow(image_array, cmap="gray")
    plt.show()
    
    

####################################
# UPDATE DATAFRAME
####################################
def get_process_image_digit_array(x_row):
    return img_process(x_row['img_num_arr'])

def get_inv_image_digit_array(x_row):
    

    negative_img_path_arr = ['20150528_131957.jpg',
    '20150528_132004.jpg','20150528_132006.jpg', '20150528_132009.jpg',
    '20150528_133251.jpg','20150528_133253.jpg','20150528_133255.jpg',
    '20150528_133257.jpg','20150528_133259.jpg','20150528_133301.jpg',
    '20150528_133303.jpg','20150528_133306.jpg','20150528_133308.jpg',
    '20150528_133311.jpg','20150528_133314.jpg','20150528_133316.jpg',
    '20150528_133319.jpg','20150528_133322.jpg','20150528_133327.jpg',
    '20150528_133330.jpg','20150528_133345.jpg','20150528_133401.jpg',
    '20150528_133404.jpg','20150528_133413.jpg','20150528_134255.jpg',
    'IMG_0130.JPG','IMG_0371.JPG','IMG_0482.JPG',
    'IMG_0883.JPG','IMG_1276.JPG','IMG_1408.JPG',
    'IMG_1482.JPG',
    'IMG_1597.JPG',
    'IMG_1710.JPG',
    'IMG_2164.JPG',
    'IMG_2374.JPG',
    'IMG_2600.JPG',
    'IMG_2800.JPG',
    'IMG_3635.JPG',
    'IMG_3739.JPG',
    'IMG_4005.JPG',
    'IMG_4232.JPG',
    'IMG_4655.JPG',
    'IMG_4892.JPG',
    'IMG_48922.JPG',
    'IMG_5191.JPG',
    'IMG_5472.JPG',
    'IMG_6134.JPG',
    'IMG_6681.JPG',
    'IMG_6905.JPG',
    'IMG_7026.JPG',
    'IMG_7292.JPG',
    'IMG_7426.JPG',
    'IMG_7470.JPG',
    'IMG_7571.JPG',
    'IMG_7713.JPG',
    'IMG_8234.JPG',
    'IMG_8915.JPG',
    'IMG_9235.JPG',
    'IMG_9415.JPG',
    'IMG_9592.JPG',
    '20150528_132000.jpg',
    '20150528_132005.jpg',
    '20150528_132007.jpg',
    '20150528_133250.jpg',
    '20150528_133252.jpg',
    '20150528_133254.jpg',
    '20150528_133256.jpg',
    '20150528_133258.jpg',
    '20150528_133300.jpg',
    '20150528_133302.jpg',
    '20150528_133305.jpg',
    '20150528_133307.jpg',
    '20150528_133309.jpg',
    '20150528_133312.jpg',
    '20150528_133315.jpg',
    '20150528_133318.jpg',
    '20150528_133320.jpg',
    '20150528_133325.jpg',
    '20150528_133328.jpg',
    '20150528_133334.jpg',
    '20150528_133351.jpg',
    '20150528_133402.jpg',
    '20150528_133406.jpg',
    '20150528_134235.jpg',
    'IMG_0111.JPG',
    'IMG_0183.JPG',
    'IMG_0390.JPG',
    'IMG_0696.JPG',
    'IMG_1203.JPG',
    'IMG_1286.JPG',
    'IMG_1481.JPG',
    'IMG_1526.JPG',
    'IMG_1610.JPG',
    'IMG_1838.JPG',
    'IMG_2185.JPG',
    'IMG_2479.JPG',
    'IMG_2765.JPG',
    'IMG_2990.JPG',
    'IMG_3651.JPG',
    'IMG_3904.JPG',
    'IMG_4135.JPG',
    'IMG_4583.JPG',
    'IMG_4773.JPG',
    'IMG_4974.JPG',
    'IMG_51912.JPG',
    'IMG_5307.JPG',
    'IMG_5879.JPG',
    'IMG_6179.JPG',
    'IMG_6732.JPG',
    'IMG_6965.JPG',
    'IMG_7267.JPG',
    'IMG_7412.JPG',
    'IMG_7468.JPG',
    'IMG_7503.JPG',
    'IMG_7652.JPG',
    'IMG_7987.JPG',
    'IMG_8558.JPG',
    'IMG_9142.JPG',
    'IMG_9347.JPG',
    'IMG_9427.JPG',
    'IMG_9940.JPG']
    
    img_num_arr = x_row['img_num_arr']
    
    dir_arr = x_row['image_path'].split('/')
    image_name = dir_arr[9]
    if image_name in negative_img_path_arr:
        img_num_arr = cv2.bitwise_not(img_num_arr)

    return img_num_arr


def get_bw_image_digit_array(x_row):
    # make it 3 layers
    img_num_bw_arr = x_row['img_process_arr']
    img_num_arr = x_row['img_num_arr']
    img_num_bw_arr_3d = np.zeros_like(img_num_arr)
    img_num_bw_arr_3d[:,:,0] = img_num_bw_arr
    img_num_bw_arr_3d[:,:,1] = img_num_bw_arr
    img_num_bw_arr_3d[:,:,2] = img_num_bw_arr
    
    return img_num_bw_arr_3d


def resize_with_distortion(row, column, new_width=400, new_height=600):
    dim = (new_width, new_height)
    # resize image
    
    resized = cv2.resize(row[column], dim, interpolation = cv2.INTER_AREA)
    return resized

def process_resize_with_distortion(imgs_arr, new_width=400, new_height=600):
    dim = (new_width, new_height)
    # resize image
    imgs_resized = np.zeros_like(imgs_arr, shape=(imgs_arr.shape[0],new_height,new_width,3))
    for idx, img_arr in enumerate(imgs_arr):
        resized = cv2.resize(img_arr, dim, interpolation = cv2.INTER_AREA)
        resized = np.expand_dims(resized, axis=0)
        imgs_resized[idx] = resized
    return imgs_resized

def class_to_array(row):        
    class_arr = np.array([int(row['class'])])
    return class_arr

