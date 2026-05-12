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
        p_x = x.clone()

    def apply_fixed(self, x):
        p_x = x.clone()

        c, h, w = self.trigger.shape

        p_x[:c, self.x_loc : self.x_loc + h, self.y_loc : self.y_loc + w] = self.trigger

    def build(self):
        # no building required for badnets
        pass