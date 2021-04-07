import os
for file in os.listdir('./pictures'):
  if file.endswith('A.jpg') or file.endswith('a.jpg'):
        size = len(file)
        label = file[:size - 5]
        os.rename('./pictures/' + file, './pictures/' + label + '.jpg') 
  if file.endswith('B.jpg') or file.endswith('b.jpg'):
    os.remove('./pictures/' + file) 