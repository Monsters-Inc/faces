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
df.drop(labels=['age'], axis=1)
df = df.drop_duplicates(subset='image')

tot_images = len(df['image'])
if len(sys.argv) > 1:
  tot_images = int(sys.argv[1])

data = []

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
    if len(processed[i]) > 1:
      continue

    for j, face in enumerate(processed[i]):
      t = face.rect.top()
      l = face.rect.left()
      r = face.rect.right()
      b = face.rect.bottom()

      l = max(0, l)
      t = max(0, t)
      r = max(0, r)
      b = max(0, b)

      dataframe_index = i + start_index
      image = images[i]
      data.append([df['image'][dataframe_index], df['gender'][dataframe_index]])
    
      name = df['image'][dataframe_index]

      cropped = image[t:b, l: r]
      cv2.imwrite(end_path + name, cropped)


frac, batches = math.modf(tot_images/batch_size)
remainder = int(frac * batch_size) + 1
batches = int(batches)

for i in range(batches):
  print('Doing ' + str(i * batch_size) + ' - ' + str((i + 1) * batch_size))
  process_batch(i * batch_size, (i + 1) * batch_size)

remainder_start = (batches) * batch_size
remainder_stop = (batches) * batch_size + remainder

print('Doing ' + str(remainder_start) + ' - ' + str(remainder_stop))

process_batch(remainder_start, remainder_stop)

df_new = pd.DataFrame(data, columns=['image', 'gender'])
df_new.to_csv('data.csv', sep=';')
