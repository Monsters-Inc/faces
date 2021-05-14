import os
import sys
from tensorflow import keras 
from data import preprocess
import numpy as np

img_shape = (96, 96, 1)

directory = sys.argv[1]

model_path = 'g_bw_final.h5'

faces = preprocess(directory, img_shape, ['median', 'gray'])

model = keras.models.load_model(model_path)

predictions = model.predict(faces)


trues = []
files = sorted(os.listdir(directory))

for file in files:
  if 'm' in file:
    trues.append(0)
  
  if 'f' in file:
    trues.append(1)


corrects = 0
labels = ['male', 'female']
for i, pred in enumerate(predictions):
  index = np.argmax(pred)
  if index == trues[i]:
    corrects+=1
  print(files[i], ' - ', labels[index])

print(f'\nPrediction Accuracy: {corrects/len(predictions)}')
