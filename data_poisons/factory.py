from .register import POISON_REGISTRY
import clean_label
import dirty_label

def data_poison_factory(cfg):

    poison_name = cfg.name

    poison_class = POISON_REGISTRY[poison_name]

    poison = poison_class(cfg)

    return poison