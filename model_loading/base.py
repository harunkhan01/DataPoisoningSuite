import torch

class GenericModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_classes = cfg['output_classes']
        self.build()