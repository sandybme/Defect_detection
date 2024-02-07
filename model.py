import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):# Net for region based segment classification
        def __init__(self,NumClasses=1):
            super(Net, self).__init__()

            self.Net = models.resnet50(pretrained=True)
            self.Net.fc=nn.Linear(2048, int(NumClasses))

        def AddAttententionLayer(self):
            self.ValeLayers = nn.ModuleList()
            self.Valve = {}
            self.BiasValve = {}
            ValveDepths = [64] # Depths of layers were attention map will be used
            for i, dp in enumerate(ValveDepths):
                self.Valve[i] = nn.Conv2d(1, dp, stride=1, kernel_size=3, padding=1, bias=True)
                self.Valve[i].bias.data = torch.zeros(self.Valve[i].bias.data.shape)
                self.Valve[i].weight.data = torch.zeros(self.Valve[i].weight.data.shape)
            for i in self.Valve:
                self.ValeLayers.append(self.Valve[i])
                
        def forward(self,Images,ROI,Trainmode=True):
        
                nValve = 0  # counter of attention layers used
                x=Images
                x = self.Net.conv1(x)
                if Trainmode:
                    AttentionMap = self.Valve[nValve](F.interpolate(ROI, size=x.shape[2:4], mode='bilinear'))
                    x = x + AttentionMap
                    nValve += 1
                else:
                    x = x
                x = self.Net.bn1(x)
                x = self.Net.relu(x)
                x = self.Net.maxpool(x)
                x = self.Net.layer1(x)
                x = self.Net.layer2(x)
                x = self.Net.layer3(x)
                x = self.Net.layer4(x)
                x = torch.mean(torch.mean(x, dim=2), dim=2)
                #x = x.squeeze()
                x = self.Net.fc(x)
                return x
