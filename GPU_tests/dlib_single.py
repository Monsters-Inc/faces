import sys
import cv2
import dlib
import numpy as np
import math
import pandas as pd

weights = 'premade.dat'
path = '../pictures/'

if len(sys.argv) < 2:
  print('image name required')
  quit()

image_path = path + sys.argv[1]

if sys.argv[1][-4:] != '.jpg':
  image_path  = image_path + '.jpg'

image = cv2.imread(image_path)

dim = (771, 1230)
image = cv2.resize(image, dim)

model = dlib.cnn_face_detection_model_v1(weights)
faces = model(image)

for face in faces:
  x = face.rect.left()
  y = face.rect.top()
  w = face.rect.right()
  h = face.rect.bottom()
  x = max(0, x)
  y = max(0, y)
  w = max(0, w)
  h = max(0, h)

  #cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
  crop = image[y:h, x:w]
  cv2.imshow(sys.argv[1], crop)
  cv2.waitKey()

#cv2.imshow(sys.argv[1], image)
#cv2.waitKey()

