import os.path
import pandas as pd # data
import seaborn as sea # plotting
import cv2 # image reading

df = pd.read_csv('../data/dataset.csv', sep=';')

hej = df.loc[df['image'] == '001.jpg']
print(hej)