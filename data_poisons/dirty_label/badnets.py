import copy

from ..base import GenericPoison
from ..register import register_poison

@register_poison('badnets')
class Badnets(GenericPoison):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def apply(self, x):
        if self.random_placement:
            self.apply_random(x)
        else:
            self.apply_fixed(x)

    def apply_random(self, x):
        pass

    def apply_fixed(self, x):
        p_x = copy.deepcopy(x)

        p_x[:, ] = 

    def build(self):
        # trigger will depend on trigger size
        self.trigger = 