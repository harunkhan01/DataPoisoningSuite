import torch
import torchvision.models as models

from ..base import GenericModel
from ..registry import register_model

@register_model('resnet50')
class ResNet50(GenericModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build(self):
        self.model = self.build_resnet50()

    def build_resnet50(self):
        self.weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=self.weights)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.output_classes)