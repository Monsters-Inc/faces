import os
import sys
from tensorflow import keras 
from data import preprocess
import numpy as np

img_shape = (96, 96, 3)
model_path = 'g_final_median.h5'
preprocessing = []
directory = 'our_dataset'

arg = sys.argv

if len(arg) > 1:
    directory = arg[1].lower()
    if len(arg) > 2:
      if arg[2] == 'gray' or arg[2] == 'he' or arg[2] == 'canny' or arg[2] == 'median':
        preprocessing.append('gray')
      
      preprocessing.append(arg[2].lower())

faces = preprocess(directory, img_shape, preprocessing)

model = keras.models.load_model(model_path)

predictions = model.predict(faces)

trues = []
files = sorted(os.listdir(directory))

count = 0
for file in files:
  if 'm' in file or 'M' in file:
    trues.append(0)
    count+=1
  
  if 'f' in file or 'F' in file or 'k' in file or 'K' in file:
    trues.append(1)
    count+=1

print(count)

corrects = 0
labels = ['male', 'female']
for i, pred in enumerate(predictions):
  index = np.argmax(pred)
  if index == trues[i]:
    corrects+=1
  print(files[i], ' - ', labels[index])

# This appends the accuracy to results.txt
f = open("results.txt", "a")
f.write(str(corrects/len(predictions))+'\n')
f.close()

print(f'\nPrediction Accuracy: {corrects/len(predictions)}')

