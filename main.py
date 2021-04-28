import os.path
import pandas as pd # data
import seaborn as sea # plotting
import cv2 # image reading

import sys

print('Input from React: ', str(sys.argv[1]))

df = pd.read_csv('../data/dataset.csv', sep=';')

person = df.loc[df['image'] == sys.argv[1]+'.jpg']
gender = person['gender'].to_numpy()[0]
print(gender)




# import argparse
# parser = argparse.ArgumentParser(description='To read a file sent to terminal?')
# parser.add_argument('filename', type='String', help='the filename to review')
# args = vars(parser.parse_args())
# print(args)