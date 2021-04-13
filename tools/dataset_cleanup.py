import pandas as pd # data
import seaborn as sea # plotting
import cv2 # image reading


df = pd.read_excel('data/ageGender.xlsx') ##på Mac ska man ej ha ../ utan direkt filen
modified = df[['ImageFrontID', 'Kon', 'Alder']]
modified.columns = ['image', 'gender', 'age']
modified['image'] = modified['image'].str.replace('A', '')



#print(modified)

#print(df[df.gender == 'K'].shape[0])

modified.to_csv('data/dataset.csv', index=False, sep=';') ##på Mac ska man ej ha ../ utan direkt filen