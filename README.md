# Digital Display Reader
Machine learning project to detect and classify 7 segment LCD characters in photos of digital displays

## Pre-requisites
To run this project, 
1. install OpenCV Contrib Python  
`pip install opencv-contrib-python`
2. install IM Utils  
`pip install --upgrade imutils`
3. Download and install Tesseract  
Linux: `apt install tesseract-ocr`  
Mac:  `brew install tesseract`  
4. Install PyTesseract  
`pip install pytesseract`
5. Install Deskew  
`pip install deskew`
6. Install Flask  
`pip install flask`  
`pip install flask-ngrok`  
`pip install flask-restful`  

## Jupyter Workbooks
* TrainingDataProcessing.ipynb - This workbook reads the training data jpg files and XML annotation files, creates a Pandas dataframe and stores it as a pickle file
* TrainClassifierModel.ipynb - This workbook runs on Google Colab and reads the pickle file of the dataframe, builds the ResNet50 model, trains the model and then exports the model to be stored on Google Drive
* Digital_reader_API.ipynb - This workbook runs on Google Colab and imports the pretrained Tesseract model, the trained ResNet50 model, and creates a Flask application with a temporary public URL

## Python files
* lhl_build_dataframe.py - Contains methods to add columns to the training dataframe needed for training the model
* lhl_image_transform.py - Contains methods used to transform images from the API to a format needed by Tesseract to determine character bounding co-ordinates
* lhl_model_utils.py - Methods to load the ResNet50 model and run predictions.
