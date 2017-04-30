import sys
import os
import time
import random
import numpy as np
import theano
import theano.tensor as T

import lasagne


class ColorPie:
    def __init__(self, dataset):
        self.dataset = dataset

    def build_sets(self):
        train_size = round(len(self.dataset) * 0.8)
        train = random.sample(self.dataset, train_size)
        val_test = list(set(self.dataset).difference(set(train)))
        val_size = round(len(self.dataset) * 0.1)
        val = random.sample(val_test, val_size)
        test = list(set(val_test).difference(set(val)))

        X_train = [d[0] for d in train]
        y_train = [d[1] for d in train]
        X_val = [d[0] for d in val]
        y_val = [d[1] for d in val]
        X_test = [d[0] for d in test]
        y_test = [d[1] for d in test]
        return X_train, y_train, X_val, y_val, X_test, y_test

