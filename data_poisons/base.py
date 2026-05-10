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

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def build(self):
        pass