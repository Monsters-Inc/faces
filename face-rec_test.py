import sys
from PIL import Image, ImageDraw
import pandas as pd # data
import seaborn as sea 
import face_recognition
import cv2

df = pd.read_csv('data/dataset.csv', sep=';')



img = face_recognition.load_image_file('pictures/' + sys.argv[1] + '.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) + 50
face_land = face_recognition.face_landmarks(img)

pil_image = Image.fromarray(img)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_land:
  for facial_feat in face_landmarks.keys():
    d.line(face_landmarks[facial_feat], width=5, fill=(255, 0, 0, 255))

pil_image.show()


#print(df.image[0])

