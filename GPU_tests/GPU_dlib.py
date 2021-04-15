import sys
import cv2
import dlib
import numpy as np
import math
import pandas as pd

start_path = '../pictures/'
end_path = '../cropped_pictures/'
weights = 'premade.dat'
batch_size = 500

df = pd.read_csv('../data/dataset.csv', sep=';')
df = df.drop_duplicates(subset='image')

print(df)
quit()

model = dlib.cnn_face_detection_model_v1(weights)

def process_batch(start_index, end_index):
  images = []
  for i in range(start_index, end_index):

    #image_path = start_path + img
    image_path = start_path + df['image'][i]
    image = cv2.imread(image_path)

    dim = (771, 1230)
    resized = cv2.resize(image, dim)

    images.append(resized)

  processed = model(images, 0, 30)

  for i in range(len(processed)):

    for j, face in enumerate(processed[i]):
      x = face.rect.left()
      y = face.rect.top()
      w = face.rect.right()
      h = face.rect.bottom()

      dataframe_index = i + start_index
      image = images[i]
      name = df['image'][dataframe_index][:-4]
      name = name + '_' + str(j) + '.jpg'
      cropped = image[y:h, x: w]

      if cropped.size > 0:
        cv2.imwrite(end_path + name, cropped)

tot_images = len(df['image'])
frac, batches = math.modf(tot_images/batch_size)
remainder = int(frac * batch_size)
batches = int(batches)

'''
for i in range(batches):
  print('Doing ' + str(i * batch_size) + ' - ' + str((i + 1) * batch_size))
  process_batch(i * batch_size, (i + 1) * batch_size)
'''

remainder_start = (batches) * batch_size
remainder_stop = (batches) * batch_size + remainder

print('Doing ' + str(remainder_start) + ' - ' + str(remainder_stop))

process_batch(remainder_start, remainder_stop)