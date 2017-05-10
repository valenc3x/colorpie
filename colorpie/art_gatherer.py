""" Art Gatherer class module """
from collections import namedtuple
from datetime import datetime
from urllib import request

import numpy as np
from mtgsdk import Card, Set

import cv2

MagicCard = namedtuple('MagicCard', [
    'card_id',
    'name',
    'set_code',
    'set_name',
    'color_identity',
    'image',
    'artwork'
])


class ArtGatherer:
    """ Wrapper class to fetch card information from mtgsdk
    """

    def __init__(self):
        pass

    @staticmethod
    def _color_to_identity(colors):
        """ Returns the color identity based on a list of card colors
            Usefull to name colorless and Multicolor cards
        """
        if not colors:
            return 'Colorless'
        if len(colors) > 1:
            return 'Multicolor'
        return colors[0]

    @staticmethod
    def _url_to_image(image_url='goo.gl/QBqtqE'):
        """ Returns a cv2 image object from a gatherer image_id
        """
        resp = request.urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def _get_artwork(card, image):
        """ Crops image to fit artwork only."""
        return image[40:165, 25:195]

    @classmethod
    def _build_magic_card(cls, card):
        identity = cls._color_to_identity(card.colors)
        image = cls._url_to_image(card.image_url)
        artwork = cls._get_artwork(card, image)
        return MagicCard(
            card.multiverse_id,
            card.name,
            card.set,
            card.set_name,
            identity,
            image,
            artwork
        )

    @classmethod
    def get_card_info(cls, card_id=40545):
        """ Returns a cv2 image object from a multiverse id of a specific card
        """
        card = Card.find(card_id)
        return cls._build_magic_card(card)

    @staticmethod
    def print_cardset_names():
        all_sets = Set.all()
        for mtgset in all_sets:
            print(mtgset.code, mtgset.name)

    @classmethod
    def get_full_set(cls, code=None):
        """ Returns a card list for a full set based on set codename
        """
        card_set = list()
        if code is None:
            return card_set
        print('Searching for cards...')
        fullset = Card.where(set=code).all()
        print('Building card set...')
        for card in fullset:
            # Skip lands. Too basic
            if 'Land' in card.types:
                continue
            magic_card = cls._build_magic_card(card)
            card_set.append((magic_card.artwork, magic_card.color_identity))
        return card_set

    @classmethod
    def get_card_list(cls, card_id_list=None):
        """ Returns card list based on a list of multiverse ids
        """
        cardlist = list()
        for cid in card_id_list:
            cardlist.append(cls.get_card_info(cid))
        return cardlist
