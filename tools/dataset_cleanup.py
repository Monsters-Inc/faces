import pandas as pd # data
import seaborn as sea # plotting
import cv2 # image reading


df = pd.read_excel('../ageGender.xlsx')
modified = df[['ImageFrontID', 'Kon']]
modified.columns = ['image', 'gender']
modified['image'] = modified['image'].str.replace('A', '')

#print(modified)

print(df[df.gender == 'K'].shape[0])

modified.to_csv('dataset.csv', index=False, sep=';')