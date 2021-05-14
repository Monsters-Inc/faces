import sys
from tensorflow import keras 
from data import preprocess
import numpy as np

img_shape = (48, 48, 3)

path = sys.argv[1]

model_path = 'g_final.h5'

faces = preprocess(path, img_shape, [] , False)

model = keras.models.load_model(model_path)

predictions = model.predict(faces)

labels = ['male', 'female']
for pred in predictions:
  print(labels[np.argmax(pred)])
