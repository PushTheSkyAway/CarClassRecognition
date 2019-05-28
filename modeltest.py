import itertools
import sys

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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')

    print(cm)

    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


img_width, img_height = 96, 96
num_of_test_samples = 8000
batch_size = 128

validation_data_dir = 'data/validation'


if K.image_data_format() == 'channel_first':
    image_shape = (3, img_width, img_height)
else:
    image_shape = (img_width, img_height, 3)

test_datagen = ImageDataGenerator()
test_data_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(
    img_width, img_height), batch_size=batch_size, shuffle=False)

if len(sys.argv)<2:
    print("Usage: python modeltest.py <model_file>")
else:
    model = load_model(sys.argv[1])


# testing
predictions = model.predict_generator(
    test_data_generator, steps=num_of_test_samples // batch_size+1)
predicted_classes = predictions > 0.5
predicted_classes = np.argmax(predicted_classes, axis=1)

true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())

report = classification_report(
    true_classes, predicted_classes, target_names=class_labels)
print(report)

print('Confusion Matrix')
cm = confusion_matrix(test_data_generator.classes, predicted_classes)
print(cm)


score = model.evaluate_generator(
    test_data_generator, steps=num_of_test_samples // batch_size+1)
print('Loss value: ', score[0])
print('Accuracy: ', score[1])


plot_confusion_matrix(cm, class_labels)
plt.show()
