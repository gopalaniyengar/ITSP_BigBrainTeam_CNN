import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

transform=transforms.ToTensor()
train=datasets.EMNIST(root='./data', split= 'byclass', train=True, download=True, transform=transform)
test=datasets.EMNIST(root='./data', split= 'byclass', train=False, download=True, transform=transform)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        #remember to account for pooling layers while initializing following tensor
        #nfilt=torch.tensor([1,1])
        #nconvlayer = len(list(nfilt))

        self.convlayers= nn.Sequential(
            #for i in range(0,nconvlayer):
             #   if i==0:
                 nn.Conv2d(1,1,kernel_size=3,stride=0,padding=1),
                 nn.ReLU(),
                 nn.BatchNorm2d(1),
                 nn.MaxPool2d(kernel_size=2, stride=2),
              #  else
                 nn.Conv2d(1,1, kernel_size=3, stride=0, padding=1),
                 nn.ReLU(),
                 nn.BatchNorm2d(1),
                 nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.fclayer= nn.Linear(49,62)

    def forward(self, x):
            x = self.convlayers(x)
            x = x.view(x.size(0), -1)
            x = self.fclayer(x)
            return x

CNN=Model()
print(CNN)
#balancingwts= torch.tensor(62)
#loss= nn.CrossEntropyLoss(weight=)
optimizer= optim.Adam(CNN.parameters(), lr=1e-5)

