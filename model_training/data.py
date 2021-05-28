import os
import cv2
import numpy as np
import pandas as pd
import dlib
from sklearn.model_selection import train_test_split
from preprocessing_tools import format_folder_name, he_single, clahe_single, canny_edges_single, median_filtering_single

face_detection_model_path = 'premade.dat'

def get_images(filenames, dir_path, image_shape, logging):
    dir_path = format_folder_name(dir_path)
    images = []
    count = 0
    for image in filenames:
        if logging:
            print('Getting Images: '+image+' | ' +
                  str(count)+'/'+str(len(filenames)), end='\r')

        temp = cv2.imread(dir_path + image, 0)
        temp = cv2.resize(temp, (image_shape[0], image_shape[1]))

        images.append(temp)
        count += 1

    return np.array(images)

def load_data(dataset, image_folder, img_shape, test_size, preprocessing = [], no_images = 0, equal = False, require_age = False, logging = True):

    df = pd.read_csv(dataset, sep=';')

    if require_age:
      df.dropna(0, 'any', inplace=True)

    if equal:
      women = df.loc[df['gender'] == 'K', ['image', 'age']]
      men = df.loc[df['gender'] == 'M', ['image', 'age']]

      no_images = no_images / 2

      if no_images == 0 or no_images >= len(women.index):
        no_images= len(women.index)

      women = women.sample(n=no_images)
      men = men.sample(n=no_images)

      df = women.append(men)
    
    elif not no_images == 0:
      df = df.sample(n = no_images)

    one_hot = pd.get_dummies(df['gender'])
    df = df.drop('gender', axis = 1)
    df = df.join(one_hot)
    df = df.rename(columns={0:'M', 1:'K'})

    train, test= train_test_split(df, test_size=test_size, shuffle=True)

    train_labels = train[['M', 'K', 'age']]
    test_labels = test[['M', 'K', 'age']]

    train_images = get_images(train['image'].values, image_folder, img_shape, logging)
    test_images = get_images(test['image'].values, image_folder, img_shape, logging)

    train_images = preprocess(train_images, img_shape, preprocessing)
    test_images= preprocess(test_images, img_shape, preprocessing)

    return train_images, test_images, train_labels, test_labels


def get_faces(image):

  model=dlib.cnn_face_detection_model_v1(face_detection_model_path)
  faces=model(image)

  cropped=[]

  for face in faces:
    x=face.rect.left()
    y=face.rect.top()
    w=face.rect.right()
    h=face.rect.bottom()
    x=max(0, x)
    y=max(0, y)
    w=max(0, w)
    h=max(0, h)

    crop=image[y:h, x:w]
    cropped.append(crop)
    # cv2.imshow('bild', crop)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

  return cropped

def preprocess(input_data, img_shape, preprocessing):
  faces=[]
  if isinstance(input_data, str):
    input_data=format_folder_name(input_data)
    folder=sorted(os.listdir(input_data))

    images = []
    for file in folder:
      img=cv2.imread(input_data + file, 0)
      images.append(img)

    for image in images:
      cropped_faces=get_faces(image)
      faces=faces + cropped_faces

  else:
    faces=input_data

  for i in range(len(faces)):
    faces[i]=cv2.resize(faces[i], (img_shape[0], img_shape[1]))

    if len(preprocessing) > 0:
      if 'he' in preprocessing:
        faces[i]=he_single(faces[i])

      if 'clahe' in preprocessing:
        faces[i]=clahe_single(faces[i])

      if 'canny' in preprocessing:
        faces[i]=canny_edges_single(faces[i])
      
      if 'median' in preprocessing:
        faces[i]=median_filtering_single(faces[i])

  return np.array(faces)

