
from ..base import GenericPoison
from ..register import register_poison

@register_poison('badnets')
class Badnets(GenericPoison):
    def __init__(self, cfg):
        super().__init__(cfg)
        