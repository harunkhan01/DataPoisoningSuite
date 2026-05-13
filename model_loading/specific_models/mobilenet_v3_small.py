import torch
import torchvision.models as models

from ..base import GenericModel
from ..registry import register_model

@register_model("mobilenet_v3_small")
class MobilenetV3Small(GenericModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build(self):
        self.mobilenet_v3_small()

    def mobilenet_v3_small(self):
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights)
        self.model.classifier[3] = torch.nn.Linear(
            self.model.classifier[3].in_features,
            self.output_classes
        )