from torchvision.models import efficientnet_b3
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.base_model = efficientnet_b3(weights="EfficientNet_B3_Weights.DEFAULT")
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 7)

    def forward(self, x):
        return self.base_model(x)
