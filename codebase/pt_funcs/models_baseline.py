from torch import nn
import torch

# from timm import create_model # ConvNext

class RegressionModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model
        self.fc = nn.Linear(1024, 1, bias=True)

        # Center bias in 1-10 distribution
        with torch.no_grad():
            self.fc.bias.fill_(5.5)

    def forward(self, x):
        x = self.vision_model(x).pooler_output
        scenic = self.fc(x)
        return scenic