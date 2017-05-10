from math import floor
import random
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import backend as K

K.set_image_dim_ordering('th')

COLORS = [
    'White',
    'Blue',
    'Black',
    'Red',
    'Green',
    'Colorless',
    'Multicolor'
]


class ColorPie:
    def __init__(self, dataset):
        self.dataset = dataset

    def build_sets(self, train_size):
        n = len(self.dataset)
        train_s = floor(n * train_size)
        val_s = floor((n - train_s)/2)
        test_s = n - train_s - val_s

        random.shuffle(self.dataset)

        train = self.dataset[0:train_s]
        val = self.dataset[train_s:train_s + val_s]
        test = self.dataset[train_s + val_s:]

        X_train = np.array([d[0] for d in train]) / 255
        n, w, h, d = X_train.shape
        X_train = X_train.reshape((n, d, h, w))
        y_train = np.array([COLORS.index(d[1]) for d in train])
        self.i_train = [d[2] for d in train]

        X_val = np.array([d[0] for d in val]) / 255
        n, w, h, d = X_val.shape
        X_val = X_val.reshape((n, d, h, w))
        y_val = np.array([COLORS.index(d[1]) for d in val])
        self.i_val = [d[2] for d in val]

        X_test = np.array([d[0] for d in test]) / 255
        n, w, h, d = X_test.shape
        X_test = X_test.reshape((n, d, h, w))
        y_test = np.array([COLORS.index(d[1]) for d in test])
        self.i_test = [d[2] for d in test]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_cnn(self, width, height, depth, classes, weightsPath=None):
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(
            depth, height, width), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        self.model = model
        return model

    def fit(self, X_test, y_test):
        model.fit()
