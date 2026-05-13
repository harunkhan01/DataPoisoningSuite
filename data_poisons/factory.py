from .register import POISON_REGISTRY
from . import clean_label
from . import dirty_label

def data_poison_factory(cfg):

    poison_name = cfg['name']

    poison_class = POISON_REGISTRY[poison_name]

    poison = poison_class(cfg)

    return poison