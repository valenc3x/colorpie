from collections import namedtuple

from colorpie.art_gatherer import ArtGatherer
from colorpie.image_processing import ShapeDetector

CardList = namedtuple('CardList', [
    'card_id',
    'name',
    'set_code',
    'set_name',
    'color_identity',
    'image'
])
