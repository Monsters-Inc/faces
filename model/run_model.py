import os
import sys
from tensorflow import keras
from data import preprocess
import numpy as np

img_shape = (96, 96, 3)
model_path = 'g_final_dataset_grayscale_testing.h5'
preprocessing = ['gray']
directory = ''

arg = sys.argv

if len(arg) > 1:
  directory = arg[1].lower()

  if len(arg) > 2:
    preprocessing.append(arg[2].lower())
    model_path = 'g_testing_'+arg[2].lower()+'.h5'


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

