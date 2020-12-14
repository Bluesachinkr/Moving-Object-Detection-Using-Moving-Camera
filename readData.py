import numpy as np
import os
from PIL import Image
os.chdir('data')

def readfolder():
    testData = []
    classes = 2
    cur_path = os.getcwd()
    for i in range(classes):
        path = os.path.join(cur_path, str(i))
        images = os.listdir(path)
        data = []
        for a in images:
            try:
                image = Image.open(path + '/' + a)
                image = np.array(image)
                data.append(image)
            except Exception as  e:
                print(e)
        testData.append(data)

    return testData