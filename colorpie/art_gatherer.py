""" Art Gatherer class module """
from collections import namedtuple
from urllib import request

import numpy as np
from mtgsdk import Card

import cv2

CardList = namedtuple('CardList', [
    'card_id',
    'name',
    'set_code',
    'set_name',
    'color_identity',
    'image'
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

    @classmethod
    def card_info(cls, card_id=40545):
        """ Returns a cv2 image object from a multiverse id of a specific card
        """
        card = Card.find(card_id)
        return CardList(
            card.multiverse_id,
            card.name,
            card.set,
            card.set_name,
            cls._color_to_identity(card.colors),
            cls._url_to_image(card.image_url)
        )

    @classmethod
    def get_full_set(cls, set_code='ORI'):
        """ Returns a card list for a full set based on set codename
        """
        fullset = Card.where(set=set_code).all()
        card_set = list()
        for card in fullset:
            if 'Land' in card.types:
                continue
            card_set.append(cls.card_info(card.multiverse_id))
        return card_set

    @classmethod
    def get_card_list(cls, card_id_list=None):
        """ Returns card list based on a list of multiverse ids
        """
        cardlist = list()
        for cid in card_id_list:
            cardlist.append(cls.card_info(cid))
        return cardlist
