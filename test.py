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
    # card_set = ArtGatherer.get_full_set(code='GPT')

    # for card in random.sample(card_set, 20):
    for card in card_set:
        cv2.imshow(card.name, ImageProcessing.resize_to_width(card.artwork, 300))
    cv2.waitKey(0)


def normalize():
    card = ArtGatherer.card_info(151137)
    normal = ImageProcessing.normalize_image(card.artwork)
    print(normal)


def color_test():
    ds = [(random.randint(10, 99), random_generator(4)) for _ in range(100)]
    cp = ColorPie(ds)
    train_val_test = cp.build_sets()
    pprint(train_val_test)


def dataset_test():
    ArtGatherer.print_cardset_names()
    selected = input('Type dataset code:  ')
    print('Getting card set')
    cardset = ArtGatherer.get_full_set(selected)
    print('Card set retieved')
    print('Building dataset')
    cp = ColorPie(cardset)
    print('Separating sets')
    X_train, y_train, X_val, y_val, X_test, y_test = cp.build_sets(0.8)
    print(X_train.shape)
    print('Dataset built')
    n, d, h, w = X_train.shape
    y_train = utils.to_categorical(y_train, num_classes=7)
    y_val = utils.to_categorical(y_val, num_classes=7)
    y_test = utils.to_categorical(y_test, num_classes=7)
    model = cp.build_cnn(width=w, height=h, depth=d, classes=7)

    model.fit(
        X_train, y_train,
        # validation_data=(X_val, y_val),
        batch_size=32, epochs=20, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Large CNN Error: %.2f%%" % (100 - accuracy * 100))
    y_results = model.predict_classes(X_test)
    print(y_test)
    print(y_results)

if __name__ == '__main__':
    # show_artwork()
    # normalize()
    # color_test()
    dataset_test()

