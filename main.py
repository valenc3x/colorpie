""" Test script to try general functionality """
import random
import cv2
import string
from pprint import pprint
from colorpie import ArtGatherer
from colorpie import ImageProcessing
from colorpie import ColorPie, COLORS
from keras import utils


def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))


def show_artwork():
    card_ids = [40545, 413581, 420645, 380426, 279712, 151137]
    card_set = ArtGatherer.get_card_list(card_ids)

    for card in card_set:
        cv2.imshow(
            card.name,
            ImageProcessing.resize_to_width(card.artwork, 300)
        )
    cv2.waitKey(0)


def dataset_test():
    ArtGatherer.print_cardset_names()
    selected = input('Type dataset code:  ')
    print('[INFO]\tGetting card set')
    cardset = ArtGatherer.get_full_set(selected)
    print('[INFO]\tCard set retieved')
    print('[INFO]\tBuilding dataset')
    print(len(cardset))
    cp = ColorPie(cardset)
    print('[INFO]\tSeparating sets')
    X_train, y_train, X_val, y_val, X_test, y_test = cp.build_sets(0.8)
    print(X_train.shape)
    print('[INFO]\tDataset built')
    print('[INFO]\tProcessing labels')
    y_predict = y_test
    y_train = utils.to_categorical(y_train, num_classes=7)
    y_val = utils.to_categorical(y_val, num_classes=7)
    y_test = utils.to_categorical(y_test, num_classes=7)
    n, d, h, w = X_train.shape
    print('[INFO]\tBuilding CNN')
    model = cp.build_cnn(width=w, height=h, depth=d, classes=7)

    print('[INFO]\tFit model')
    model.fit(
        X_train, y_train,
        batch_size=32, epochs=20, verbose=1)

    print('[INFO]\tEvaluate model')
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print("[INFO]\tCNN Evaluation: %.2f%%" % (100 - accuracy * 100))

    print('[INFO]\tPredict labels')
    y_results = model.predict_classes(X_test)
    print(y_predict)
    print(y_results)

if __name__ == '__main__':
    # show_artwork()
    # normalize()
    # color_test()
    dataset_test()

