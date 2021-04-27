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

model = dlib.cnn_face_detection_model_v1(weights)
data = []

def getKey():
  key = cv2.waitKey(0)
  cv2.destroyAllWindows()

  if key== ord('m'):
    return 'M'
  else:
    return 'K'



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

    dataframe_index = i + start_index

    for j, face in enumerate(processed[i]):
      l = face.rect.left()
      t = face.rect.top()
      r = face.rect.right()
      b = face.rect.bottom()

      l = max(0, l)
      t = max(0, t)
      r = max(0, r)
      b = max(0, b)

      image = images[i]
      cropped = image[t:b, l: r]
      name = df['image'][dataframe_index][:-4]
      name = name + '_' + str(j) + '.jpg'

      isnull = df['gender'].isnull()[dataframe_index]
      gender = df['gender'][dataframe_index]

      if isnull:
        temp = image.copy()
        marked = cv2.rectangle(temp, (l, t), (r, b), (0, 255, 0), 2)
        cv2.imshow(name, marked)
        gender = getKey()
        print('Using gender: ' + gender)

      data.append([name, gender, df['age'][dataframe_index]])
      cv2.imwrite(end_path + name, cropped)


tot_images = len(df['image'])
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

df_new = pd.DataFrame(data, columns=['image', 'gender', 'age'])
df_new.to_csv('data.csv', sep=';')
