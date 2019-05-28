import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.client import device_lib


def main():

    TRY_NUMBER = 40

    img_width, img_height = 96, 96
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'

    nb_train_samples = 22000
    nb_validation_samples = 8000

    epochs = 300
    batch_size = 128

    if K.image_data_format() == 'channel_first':
        image_shape = (3, img_width, img_height)
    else:
        image_shape = (img_width, img_height, 3)

#train_datagen = ImageDataGenerator(
 #       rescale=1./255,
  #      shear_range=0.3,
   #     zoom_range=0.3,
    #    horizontal_flip=True
    #)

   # test_datagen = ImageDataGenerator(rescale=1./256)

    train_generator = ImageDataGenerator().flow_from_directory(train_data_dir, target_size=(
        img_width, img_height), batch_size=batch_size, class_mode='categorical', shuffle=True)
    validation_generator = ImageDataGenerator().flow_from_directory(validation_data_dir, target_size=(
        img_width, img_height), batch_size=batch_size, class_mode='categorical', shuffle=True)

    test_generator = ImageDataGenerator().flow_from_directory(validation_data_dir, target_size=(
        img_width, img_height), batch_size=batch_size, class_mode='categorical', shuffle=False)

    # building model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=image_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(5))     #Out = car, truck, bus, single-track, pedestrian
    model.add(Activation('softmax'))

    adam = optimizers.adam()

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()

    # model fitting and save
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples//batch_size, epochs=epochs,
                                  validation_data=validation_generator, validation_steps=nb_validation_samples//batch_size, callbacks=[es], use_multiprocessing=False, workers=8)

    model.save_weights('model+weights/CUDA_W'+str(TRY_NUMBER)+'.h5')
    model.save('model+weights/MODEL_C'+str(TRY_NUMBER)+'.h5')

    # test prediction
    predictions = model.predict_generator(
        test_generator, steps=nb_validation_samples // batch_size+1)

    predicted_classes = predictions > 0.5
    predicted_classes = np.argmax(predicted_classes, axis=1)

    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    report = classification_report(
        true_classes, predicted_classes, target_names=class_labels)
    print(report)

    print('Confusion Matrix')
    cm = confusion_matrix(test_generator.classes, predicted_classes)
    print(cm)

    score = model.evaluate_generator(
        test_generator, steps=nb_validation_samples // batch_size+1)
    print('Loss value: ', score[0])
    print('Accuracy: ', score[1])

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc+loss/AccCUDA'+str(TRY_NUMBER)+'.png')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc+loss/LossCUDA'+str(TRY_NUMBER)+'.png')
    plt.show()


if __name__ == '__main__':
    main()
