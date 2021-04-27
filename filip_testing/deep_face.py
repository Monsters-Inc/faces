from deepface import DeepFace
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt


sample_size = 500 

path = 'cropped_pictures/'
df = pd.read_csv('full_dataset.csv', sep=';')

models = {}
models['age'] = DeepFace.build_model('Age')
models['gender']= DeepFace.build_model('Gender')

women = df.loc[df['gender'] == 'K', ['image', 'gender', 'age']]
men = df.loc[df['gender'] == 'M', ['image', 'gender', 'age']]

women = women.sample(n=sample_size)
men = men.sample(n=sample_size)

tot = women.append(men)
tot = tot.sample(frac=1)
#tot['gender'] = tot['gender'].map({'K': 1, 'M': 0})

correct_gender = 0
mean_error = 0
tot_ages = 0

for index, row in tot.iterrows():

  img_path = path + row['image']
  out = DeepFace.analyze(img_path = img_path, actions=['age', 'gender'], models=models, enforce_detection=False, detector_backend='dlib')
  predicted_gender = ''

  '''
  print(out['gender'])
  temp = cv.imread(img_path)
  cv.imshow('memes', temp)
  cv.waitKey(0)
  cv.destroyAllWindows()
  '''


  if out['gender'] == 'Man':
    predicted_gender = 'M'
  else:
    predicted_gender = 'K'

  if predicted_gender == row['gender']:
    correct_gender += 1

  if not pd.isna(row['age']):
    mean_error += abs(row['age'] - out['age'])
    tot_ages += 1

mean_error = mean_error/tot_ages

print('Mean age error: ' + str(mean_error))
print('Correct gender classifications: ' + str(correct_gender) + '/' + str(len(tot['image'])))



