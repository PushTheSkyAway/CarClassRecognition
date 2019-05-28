import itertools
import sys
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix


def load_image(img_path, show=False):
    img_width, img_height = 96, 96
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)                   
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img2 = img_tensor/255.                                    

    if show:
        plt.imshow(img2[0])                      
        plt.axis('off')
        plt.show()

    return img_tensor



if len(sys.argv) < 3:
    print("Usage: predict <model used> <folder with images>")


path = sys.argv[2]
model_name  = sys.argv[1]



model = load_model(model_name)

adam = optimizers.adam()
model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy']) 


files = []

for (dirpath, dirnames, filenames) in os.walk(sys.argv[2]):
    files.extend(filenames)
    break


labels=['car','truck','single-track','bus','pedestrian']

photo = []
prediction = []

for img in files:
    photo = load_image(path+'/'+img, show=True)
    prediction = model.predict(photo)
    pred_class = prediction.argmax(axis=-1)
    pred_class = sorted(labels)[pred_class[0]]
    print(str(pred_class).upper())




