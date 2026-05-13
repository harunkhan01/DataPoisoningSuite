from .registry import DATASET_REGISTRY
from . import datasets

def data_loading_factory(cfg):

    dataset_name = cfg['name']

    dataset_class = DATASET_REGISTRY[dataset_name]

    dataset = dataset_class(cfg)

    return dataset