"""
Generic Poison implements two special methods
    1. Apply - calls the subclass function that applies a poison to a particular image
    2. Build - in the case a poison needs to be constructed (inherent triggers) this function generates the 
        poison when called
"""
import abc

class GenericPoison(abc.ABC):
    def __init__(self, cfg):
        self.tgt_lbl = cfg.target_label
        self.poison_ratio = cfg.poison_ratio
        self.trigger_size = cfg.trigger_size
        self.random_placement = True if cfg.random_placement == 1 else False
        self.poison_loc = (cfg.x_loc, cfg.y_loc) if not self.random_placement else None

    @abstractmethod
    def apply(self):
        # apply the trigger pattern to a sample
        pass

    @abstractmethod
    def build(self):
        # construct the trigger pattern
        pass