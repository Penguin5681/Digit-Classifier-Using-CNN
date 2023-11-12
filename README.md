# Digit Classifier Using CNN
A Simple Digit Classification program made using Convolutional Neural Network 

This project involves building a Convolutional Neural Network (CNN) to classify digits. The model is trained using image data representing various digits and utilizes techniques from deep learning and image processing.

## Overview

The project involves:

- Preprocessing the image data by resizing, converting to grayscale, and applying histogram equalization.
- Implementing a CNN architecture using the Keras deep learning framework.
- Augmenting the training data for better model generalization.
- Training the model on a dataset of digit images.
- Evaluating the trained model's performance on a test dataset.

## Methods Used

- **Libraries**: NumPy, OpenCV (cv2), os, Keras, Matplotlib, and Pickle.
- **Data Handling**: Utilizing train-test split and generating training, testing, and validation sets.
- **Image Preprocessing**: Converting images to grayscale, equalizing histograms, and normalization.
- **Data Augmentation**: Using `ImageDataGenerator` for augmenting training data.
- **Model Architecture**: Utilizing Conv2D, MaxPooling2D, Dropout, Dense layers in the CNN.
- **Model Compilation**: Using Adam optimizer, categorical crossentropy loss, and accuracy metrics.
- **Visualization**: Plotting bar graphs for image class distributions and loss/accuracy trends.
- **Serialization**: Saving the trained model using Pickle for future use.

## Usage

1. Ensure all necessary libraries are installed (Keras, NumPy, OpenCV, Matplotlib, etc.).
2. Set up the dataset in a directory named 'myData' containing subdirectories for each class of digit images.
3. Run the Python script provided to train the model and evaluate its performance.


