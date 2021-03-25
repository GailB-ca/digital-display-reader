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
import pandas as pd

def img_process(img_array, 
                blur=True,
                blur_kernel=(193,193),
                gray=True, 
                flatten=True, 
                close_erode=True, 
                flatten_2=True,
                threshold_block_size=91,
                erode_iterations=2,
                erode_kernal=(10,10)
               ):
    
    # Threshold block size should be lower values for noisey images
    # Erode iterations should be higher for noisey images
    
    image = img_array.copy()
    
    if(blur):
        image = cv2.GaussianBlur(image.copy(), blur_kernel, 0)
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


def compute_bounding(img_array, cmd_path):
    pytesseract.pytesseract.tesseract_cmd = cmd_path
    boxes = pytesseract.image_to_boxes(img_array, 
                                  # config='--psm 11 --oem 3 -c tessedit_char_whitelist=123456789'
                                   config='--psm 11 --oem 3'
                                  )
    #print(boxes)
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
    
    
def extract_bounded_image_arr(img_array, boxes):
    
    image = img_array.copy()
    (h, w) = image.shape[:2]
    image_area = h*w
    min_box_area = h*w*0.0035
    print("Minimum bounding area is: ", min_box_area)
    box_list = boxes.splitlines()
    sub_img_arr = [None] * len(box_list)
    box_area_arr = [None] * len(box_list)
    
    #arr = numpy.array(lst)
    
    for idx, b in enumerate(box_list):
        #print(b)
        b = b.split(' ')
        x_start = int(b[1])
        y_start = h - int(b[2])
        x_end = int(b[3])
        y_end = h - int(b[4])
        
        box_h = y_start - y_end
        box_w = x_end - x_start
        #print("area of the box is", box_h*box_w)
        
        #print("{} {} {} {}".format(x_start, x_end, y_end, y_start))
        #print(image.shape)
        num_img = image[y_end-30:y_start+30, x_start-30:x_end+30]
        if (box_w*1.4 < box_h) and (box_h*box_w > min_box_area):
            print("Adding to the array: width", box_w)
            print("Adding to the array: height", box_h)
            sub_img_arr[idx] = num_img
            box_area_arr[idx] = box_h*box_w            
        else:
            sub_img_arr[idx] = num_img
            box_area_arr[idx] = 0
            
    # Get the top n boxes by area
    np_area = np.array(box_area_arr) 
    #area_above = np.quantile(np_area, 0.25)
    
    #print("Area must be more than", area_above)
    indexes_above = np.argwhere(np_area > 0)
    
    #print(indexes_above.flatten())
    a_series = pd.Series(sub_img_arr)
     
    
    ###################
    #return sub_img_arr
    # Return the images with indices in the calculated list????
    #return sub_img_arr[indexes_above.flatten()]
    ##################
    return list(a_series[indexes_above.flatten()])
    
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


####################################
# UPDATE DATAFRAME
####################################
def get_process_image_digit_array(x_row):
    return lhl.img_process(x_row['img_num_arr'])

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
