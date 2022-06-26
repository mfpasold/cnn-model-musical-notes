import tensorflow as tf

from tensorflow import keras

from keras import datasets, layers, models
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import constants


class dataloader(tf.keras.utils.Sequence):

    def __init__(self, batch_size, img_width, img_height, data):
        self.bs = batch_size
        self.h = img_height
        self.w = img_width
        self.path = data['path'].values
        self.label = data['label'].values

    def __len__(self):
        return len(self.path) // self.bs

    def __getitem__(self, idx):
        i = idx * self.bs
        batch_paths = self.path[i: i + self.bs]
        batch_labels = self.label[i: i + self.bs]

        X = np.zeros((self.bs, self.h, self.w, 3), dtype="float32")
        y = np.zeros((self.bs, 5), dtype="int32")

        for j in range(self.bs):
            img = load_img(batch_paths[j], color_mode="rgb", target_size=(self.h, self.w))  # color_mode = "grayscale"
            img = np.array(img, dtype='float32')
            img = 1 - img / 127.5
            X[j] = img
            y[j, batch_labels[j]] = 1
        return X, y


def create_neural_model():
    df = pd.DataFrame(columns=['path', 'label'])

    for dirname, _, filenames in os.walk('.\\datasets\\datasets'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            name = dirname.split('\\')[-1]
            label = label_code[name]
            df = pd.concat([df, pd.DataFrame.from_records([{'path': path, 'label': label}])])

    df.head()

    train, test = train_test_split(df, test_size=0.2, random_state=77)

    train.head()
    test.head()

    train_gen = dataloader(5, 224, 224, train)
    test_gen = dataloader(5, 224, 224, test)

    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation=constants.ACTIVATION_RELU, input_shape=(224, 224, 3)))
    model.add(Conv2D(4, (4, 4), activation=constants.ACTIVATION_RELU))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(4, (3, 3), activation=constants.ACTIVATION_RELU))
    model.add(Conv2D(3, (3, 3), activation=constants.ACTIVATION_RELU))
    model.add(Conv2D(3, (3, 3), activation=constants.ACTIVATION_RELU))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation=constants.ACTIVATION_RELU))
    model.add(Dense(5, activation=constants.ACTIVATION_SOFTMAX))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    history = model.fit(train_gen, epochs=20, validation_data=test_gen, verbose=1)
    model.save(constants.MODEL_NAME)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    return model


def load_neural_model():
    return keras.models.load_model(constants.MODEL_NAME)


def activate_cnn_model():
    if os.path.exists(constants.MODEL_NAME):
        return load_neural_model()
    else:
        return create_neural_model()


def import_img():
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    Tk().withdraw()
    path = askopenfilename()
    img = load_img(path, color_mode="rgb", target_size=(224, 224))  # (h ,w) color_mode = "grayscale"
    img = np.array(img, dtype='float32')
    img = 1 - img / 127.5

    return img


def do_prediction():
    y1 = model.predict(imgPredict)
    s = np.argmax(y1)
    return label_decode[s]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    label_code = {'Eight': 0, 'Half': 1, 'Quarter': 2, 'Sixteenth': 3, 'Whole': 4}
    label_decode = ['Eight', 'Half', 'Quarter', 'Sixteenth', 'Whole']

    model = activate_cnn_model()

    img = import_img()

    imgPredict = img.reshape(1, 224, 224, 3)

    prediction = do_prediction()

    plt.imshow(img)
    plt.title(prediction)
    plt.show()




