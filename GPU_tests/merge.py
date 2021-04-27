import pandas as pd
import numpy as np

df = pd.read_csv('../data/dataset.csv', sep=';')    
dupes = pd.read_csv('duplicates.csv.csv', sep=';') 
fixed = pd.read_csv('./data.csv', sep=';')   

df.set_index('image')
fixed.set_index('image')
dupes.set_index('image')

df.update(fixed)
df.update(dupes)


df.reset_index(inplace=True)

print(df)

#df['gender'] = np.where(df.image.values == fixed.image.values, fixed.gender.values, df.gender.values)

#final = df

df.to_csv('test.csv', sep=';')