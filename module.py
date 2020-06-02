import torch
from torch import nn

class Model(Module)
    def __init__(self):
        super(Model,self).__init__()
        #remember to account for pooling layers while initializing following tensor
        nfilt=torch.tensor([1,1,1,1,1,1,1])
        nconvlayer = nfilt.size()

        self.convlayers= Sequential\
            (
            for i in nconvlayer:
                if i==0:
                 Conv2d(1,nfilt[i],kernel_size=3,stride=0,padding=1)
                 ReLU()
                 MaxPool2d(kernel_size=2, stride=2)
                else
                 Conv2d(nfilt[i-1], nfilt[i], kernel_size=3, stride=0, padding=1)
                 ReLU()
                 MaxPool2d(kernel_size=2, stride=2)
            )

        self.fclayer=Sequential\
            (
                Linear(28*28*nfilt2,62)
            )

    def forward(self, x):
            x = self.convlayers(x)
            x = x.view(x.size(0), -1)
            x = self.fclayer(x)
            return x

    model = Model()
    optimizer = Adam(model.parameters(), lr=1e-7)
    loss = MSELoss()
