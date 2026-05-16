import torch
import random

from ..base import GenericPoison
from ..register import register_poison

@register_poison('badnets')
class Badnets(GenericPoison):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def apply(self, x):
        if self.random_placement:
            return self.apply_random(x)
        
        return self.apply_fixed(x)

    def apply_random(self, x):
        p_x = x.clone()

        c, h, w = self.trigger.shape

        ic, ih, iw = p_x.shape

        x_loc = random.randint(0, ih - h)
        y_loc = random.randint(0, iw - w)

        p_x[:c, x_loc : x_loc + h, y_loc : y_loc + w] = self.trigger

        return p_x

    def apply_fixed(self, x):
        p_x = x.clone()

        c, h, w = self.trigger.shape

        p_x[:c, self.x_loc : self.x_loc + h, self.y_loc : self.y_loc + w] = self.trigger

        return p_x

    def build(self):
        # no building required for badnets
        pass