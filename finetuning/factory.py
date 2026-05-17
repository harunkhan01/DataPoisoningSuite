import torch

from .finetuning_class import FinetuneClass    

def fine_tuning_factory(model, train, test, device, cfg):

    ft_object = FinetuneClass(device, cfg)

    ft_object.train(model, train, test)