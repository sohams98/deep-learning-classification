# Classification of Identification Documents using Deep Learning

## Objective

The objective of this project is to demonstrate classification and machine learning techinques to classify identity documents sampled from the MIDV-500 Dataset.  
The soruce code of this repository will showcase the following skills,  
1. Data cleaning and preparation.
2. Labelling and annotatin data for classification (labelstudio).
3. Building a CNN model to train on raw data using pytorch. 
4. Building a YoloV8 model to improve classification.
5. Combining visual features based classifiers such as CNN and text strings present on the document to create a fusion classifier.

## Problem Statement

1. Classify 10 types of identity documents in their correct classes. 
2. Deploy the model as a docker image. 


## Overview 

- A CNN model was built using python (pytorch) on raw data to get a baseline. 
- After this, data was preprocessed using an object detection model (yolov8) to crop out cards from the entire image. For this, images were annotated and labelled using LabelStudio to conform to the yolo training format.
- This cropped data was fed back into the same CNN model which led to a significant improvement in accuracy. 
- A text features based model was introduced to enhance the output of the CNN model. This text feature model relies on raw OCR data from the images as well as features engineered such as spelling corrections and normalisations. 
- Finally, the outputs of the CNN as well as the text features model were combined to create a fusion model. 

A complete walkthrough of the process is present in the "Classification Case Study.ipynb". 
Note: I recommend running Cells 2 and 29 to recreate the training data used for this project. 

The final trained models are present under the "models" directory. A test_single.py and test_model.py script is provided for easy testing of the models on a single or a batch of data respectively.

Here is an outline of all the scripts in the directory,
model.py - Convolution model and data loaders.
random_sample.py - Create a random sample from the dataset.
train_baseline.py - Train baseline convolution model.
train_card.py - Train yolov8 model.
train_cropped.py - Train convolution model on cropped images.
train_text_features.py - Clean text features and train machine learning model.
crop_cards.py - Utilize the trained yolov8 model to crop cards from dataset.
fusion_model.py - Create a fusion model from cnn and text features model.
test_model.py - Test baseline and crop models using data loaders.
test_single.py - Test baseline or crop model on a single input image.

# Dockerisation
After the training the fusion model, the model along with the required code is converted into a docker image. This image can then be run on any cloud computing (like AWS ECS/EKS) or on-prem servers which support a docker daemon. 

The docker file for the same can be found under the docker folder in this repo. 
