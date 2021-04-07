import pandas as pd # data
import seaborn as sea # plotting
import cv2 # image reading


df = pd.read_excel('ageGender.xlsx')
modified = df[['ImageFrontID', 'Kon']]
modified.columns = ['image', 'gender']
modified['image'] = modified['image'].str.replace('A', '')
modified['gender'] = modified['gender'].str.replace('K', 'F')

#print(modified)

print(modified[modified.gender == 'F'].shape[0])
print(modified[modified.gender == 'M'].shape[0])

modified.to_csv('dataset.csv', index=False, sep=';')