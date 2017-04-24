""" Test script to try general functionality """
import random
import cv2
from colorpie import ArtGatherer


# cards = ArtGatherer.get_card_list([40545, 413581, 420645, 380426, 417766])
card_set = ArtGatherer.get_full_set('GPT')

for card in random.sample(card_set, 20):
    cv2.imshow(card.color_identity, card.artwork)
cv2.waitKey(0)
