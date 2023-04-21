
import torchvision
import torch
from torchsummary import summary
import pdb

class get_model(torch.nn.Module):
    def __init__(self,num_classes):
        super(get_model,self).__init__()
        self.num_classes = num_classes
        self.model =torchvision.models.efficientnet_b0(pretrained = True,weights = torchvision.models.EfficientNet_B0_Weights) # weights
        self.model.classifier[1]=torch.nn.Linear(1280,self.num_classes)
        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self,images):
        return self.model(images)
  