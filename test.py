from pprint import pprint
from colorpie import ArtGatherer
from colorpie import ShapeDetector


cards = ArtGatherer.get_card_list([40545, 413581, 420645, 380426, 417766])
pprint(cards)

# cset = ArtGatherer.get_full_set('ORI')
# pprint(cset)
