"""
Generic Poison implements two special methods
    1. Apply - calls the subclass function that applies a poison to a particular image
    2. Build - in the case a poison needs to be constructed (inherent triggers) this function generates the 
        poison when called
"""
import abc
import numpy as np

class GenericPoison(abc.ABC):
    def __init__(self, cfg):
        self.tgt_lbl = cfg.target_label
        self.poison_ratio = cfg.poison_ratio
        self.process_trigger_cfg(cfg)

    def process_trigger_cfg(self, cfg):
        cfg_trigger = cfg.trigger

        self.random_placement = cfg_trigger.placement.random

        if self.random_placement:
            self.location = None
        else:
            self.x = cfg_trigger.placement.x_loc
            self.y = cfg_trigger.placement.y_loc

        self.trigger = (
            np.array(cfg_trigger.pattern, dtype=np.uint8)
            if "pattern" in cfg_trigger
            else None
        )
        self.trigger = np.transpose(self.trigger, (2, 0, 1))

    @abc.abstractmethod
    def apply(self):
        # apply the trigger pattern to a sample
        pass

    @abc.abstractmethod
    def build(self):
        # construct the trigger pattern
        pass