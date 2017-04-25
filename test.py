""" Test script to try general functionality """
import random
import cv2
from colorpie import ArtGatherer

# card_ids = [40545, 413581, 420645, 380426, 279712, 151137]
# card_set = ArtGatherer.get_card_list(card_ids)
card_set = ArtGatherer.get_full_set(code='GPT')

# for card in card_set:
for card in random.sample(card_set, 20):
    cv2.imshow(card.name, card.artwork)
cv2.waitKey(0)
