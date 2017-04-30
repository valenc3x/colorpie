""" Test script to try general functionality """
import random
import cv2
import string
from pprint import pprint
from colorpie import ArtGatherer
from colorpie import ImageProcessing
from colorpie import ColorPie


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


if __name__ == '__main__':
    # show_artwork()
    # normalize()
    color_test()
