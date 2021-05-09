import numpy as np
import os
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from model import train_model
from data import data, data_final, data_final_other
from sklearn.preprocessing import OneHotEncoder

# Settings
image_folder = "resized_96_equal_distribution_pictures/"
dataset = "../data/full_dataset.csv"
test_size = 0.25
img_shape = (96, 96, 3)
logging = False
model_save = 'final.h5'
batch_size = 64
epochs = 500


X_train, X_test, y_train, y_test = data_final_other(dataset, image_folder, img_shape, test_size, logging)

# Train a new model
model = train_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, model_save)

print('Evaluating model: ')
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])



