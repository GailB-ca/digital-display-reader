from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import pickle
import numpy as np
from matplotlib import pyplot as plt

def plot_images_from_df(range_min, range_max, df, column_name):
    for i in range(range_min,range_max):
        image = df.loc[i][column_name]
        plt.imshow(image, cmap="gray")
        plt.show() 
    
def load_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    
    return loaded_model
    
def predict_digits(digit_imgs, model):
    
    pred_datagen = ImageDataGenerator(
        rescale=1/255.0,
        samplewise_center=True 
    )
    
    input = digit_imgs
    X_input = tf.keras.applications.resnet50.preprocess_input(input)
    max_loop = X_input.shape[0]
    loop_count = 0
    predictions = np.empty([max_loop], dtype=str)
    for batch in pred_datagen.flow(X_input, batch_size=1, shuffle=False):
        if (loop_count >= max_loop):
            break
        prediction = model.predict(batch)
        prediction_max = np.argmax(prediction, axis=1)
        prediction_perc_max = np.amax(prediction)
        print("Max probability is: ", prediction_perc_max)
        if (prediction_perc_max < 0.9500):
          # The model is unsure, so return a blank space
          predictions[loop_count] = ' '
        else:
          predictions[loop_count] = str(prediction_max[0])

        
        loop_count += 1

    return predictions