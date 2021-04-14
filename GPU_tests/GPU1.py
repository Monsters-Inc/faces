import sys
import math
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd # data
import face_recognition
import gc

df = pd.read_csv('data/dataset.csv', sep=';')

temp = []
full_length = len(df['image'])
group_size = 1
frac, groups = math.modf(full_length/group_size)
groups = int(groups)
final = int(frac * group_size)

for group in range(groups):
  images = []
  for i in range(group_size):
    curr = face_recognition.load_image_file('pictures/' + df['image'][i + (group * group_size)])
    images.append(curr)

  face_locs = []
  for i in range(len(images)):
    face_loc = face_recognition.face_locations(images[i] ,number_of_times_to_upsample=0, model='cnn')
    face_locs.append(face_loc)

  for image_no , faces in enumerate(face_locs):
    no_faces = len(face_locs)

    for face in faces:
      top, right, bottom, left = face
      face_image = images[image_no][top:bottom, left:right]

      pil_image = Image.fromarray(face_image)
      prefix = str(image_no + group * group_size)
      pil_image.save('cropped_pictures/' + prefix + '.jpg')
  #gc.collect()

#print(df.image[0])

