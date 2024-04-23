from torch import nn

# from timm import create_model # ConvNext

class ConvNext_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.basenet = create_model("convnext_large", pretrained=True)
        self.basenet.head.fc = nn.Identity()
        self.fc = nn.Linear(1536, 1, bias=False)
        # with torch.no_grad():
        #     self.fc.bias.fill_(4.5)

    def forward(self, x):
        x = self.basenet(x)
        scenic = self.fc(x) + 4.43 # Fixed bias, center of dataset
        return scenic