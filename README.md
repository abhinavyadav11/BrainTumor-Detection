# Brain Tumor Detection Model
This repository contains a deep learning model designed to detect brain tumors from MRI images. The model has been trained on a dataset of labeled MRI scans and is capable of classifying images as either containing a tumor or not. This project is aimed at helping in the early diagnosis and treatment of brain tumors.


# Overview
Brain tumors are abnormal growths in the brain that can be life-threatening. Early detection of brain tumors is crucial for successful treatment. This model leverages a Convolutional Neural Network (CNN) to classify MRI images as either having a tumor or not. The model is trained using Keras and TensorFlow.

# Model Architecture
The model is based on a Convolutional Neural Network (CNN) and has been trained using the following architecture:

Input Layer: 224x224 RGB images
Backbone: ResNet152V2 (pre-trained on ImageNet, with frozen layers)
Classification Layers: Global Average Pooling, Dense layers with ReLU and sigmoid activations
Output: Binary classification (Tumor/No Tumor)
# Dataset
The dataset used for training consists of labeled MRI images, categorized into two classes:

Tumor
No Tumor
The images were preprocessed to a uniform size of 224x224 pixels and normalized before being fed into the model.

# Installation
To use this model, clone the repository and install the required dependencies:

git clone https://github.com/abhinavyadav11/BrainTumor-Detection.git
cd brain-tumor-detection
pip install -r requirements.txt
Usage
1. Running Predictions
To run predictions on a new MRI image:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# Model Download


You can download the pre-trained model from the following link:

https://www.dropbox.com/scl/fi/hu09c1uvkhwtssa51ko8y/BTDetectionOptimized-88.h5?rlkey=6a6g4xbzf4pzybewgn58vpbaq&st=lkqpr8jt&dl=0

# Load the model
model = load_model('path_to_your_model.h5')

# Preprocess the image
img = image.load_img('path_to_image.jpg', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions

predictions = model.predict(img_array)
predicted_class = (predictions[0] > 0.5).astype("int32")

if predicted_class == 1:
    print("Prediction: Tumor")
else:
    print("Prediction: No Tumor")
#  Training the Model
If you want to train the model from scratch, use the following command:

python train.py --data_dir path_to_dataset --epochs 50



# Results
The model has achieved the following metrics on the test set:

Accuracy: 94% 
Precision: 93%
Recall: 95%
F1-Score: 94% /
These results demonstrate the model's ability to accurately detect brain tumors from MRI images.

# Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Any improvements or bug fixes are welcome!

# License
This project is licensed under the MIT License - see the LICENSE file for details.

