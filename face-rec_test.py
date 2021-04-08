import sys
from PIL import Image, ImageDraw
import pandas as pd # data
import seaborn as sea 
import face_recognition

df = pd.read_csv('data/dataset.csv', sep=';')

num = int(sys.argv[1])
img = face_recognition.load_image_file('pictures/' + sys.argv[1] + '.jpg')
face_land = face_recognition.face_landmarks(img)

pil_image = Image.fromarray(img)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_land:
  for facial_feat in face_landmarks.keys():
    d.line(face_landmarks[facial_feat], width=5)

pil_image.show()


#print(df.image[0])

