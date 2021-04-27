import pandas as pd
import tensorflow as tf
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

sample_size = 500

def get_images(paths):
  images = []
  for image in paths:
    temp = cv.imread(img_path + image)
    temp = cv.resize(temp, (64, 64))
    images.append(temp)

  return np.asarray(images)


img_path = './cropped_pictures/'

df = pd.read_csv('full_dataset.csv', sep=';')

women = df.loc[df['gender'] == 'K', ['image', 'gender']]
men = df.loc[df['gender'] == 'M', ['image', 'gender']]

women = women.sample(n=sample_size)
men = men.sample(n=sample_size)

tot = women.append(men)
tot = tot.sample(frac=1)

tot['gender'] = tot['gender'].map({'K': 1, 'M': 0})

X_train, X_test, y_train, y_test = train_test_split(
    tot.image.values, tot.gender.values, test_size=0.33)

train_images = get_images(X_train)
train_images = train_images / 255.0
test_images = get_images(X_test)
test_images = test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, y_train, epochs=50, validation_data=(test_images, y_test))


'''
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
'''

test_loss, test_acc = model.evaluate(test_images,  y_test, verbose=2)