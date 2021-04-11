import sys
from PIL import Image, ImageDraw
import pandas as pd # data
import face_recognition
import cv2

df = pd.read_csv('data/dataset.csv', sep=';')

img = face_recognition.load_image_file('pictures/' + sys.argv[1] + '.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) + 50
face_loc = face_recognition.face_locations(img,number_of_times_to_upsample=0, model='cnn')
print(face_loc)

for face in face_loc:
  top, right, bottom, left = face
  face_image = img[top:bottom, left:right]


  pil_image = Image.fromarray(face_image)
  pil_image.show()



#print(df.image[0])

