import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import sklearn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

transform=transforms.ToTensor()
train=datasets.EMNIST(root='./data', split= 'byclass', train=True, download=True, transform=transform)
test=datasets.EMNIST(root='./data', split= 'byclass', train=False, download=True, transform=transform)
print(len(train))
print(len(test))
print(len(train)+len(test))

batch= 30
trainload=DataLoader(train,batch_size=batch,shuffle=True)
testload=DataLoader(test,batch_size=batch,shuffle=True)

"""
RUN THIS CODE ONCE TO LOAD CLASS WEIGHTS TO USE IN LOSS FUNCTIONS INTO lossweights.pt 
THEN USE THIS FILE TO LOAD CLASS WEIGHTS IN ORDER TO PREVENT RUNNING THE SAME FILE AGAIN AND AGAIN

labelinsample=[]
labelset=np.zeros(62)
for i in range(len(labelset)):
    labelset[i]=i+labelset[i]
#print(labelset)

trainloadorig=DataLoader(train,shuffle=True)

for i,(x,y) in enumerate(trainloadorig):
    labelinsample.append(y[0])
labelinsample=np.array(labelinsample)

losswt=sklearn.utils.class_weight.compute_class_weight('balanced', labelset, labelinsample)
print(losswt)
print(len(losswt))
losswt=torch.from_numpy(losswt)
torch.save(losswt, 'lossweights.pt')
"""
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.convlayers= nn.Sequential(
                 nn.Conv2d(1,16,kernel_size=3,stride=0,padding=1),
                 nn.ReLU(),
                 nn.BatchNorm2d(16),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(16,32, kernel_size=3, stride=0, padding=1),
                 nn.ReLU(),
                 nn.BatchNorm2d(32),
                 nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.fclayer= nn.Linear(49*32,62)

    def forward(self, x):
            x = self.convlayers(x)
            x = x.view(x.size(0), -1)
            x = self.fclayer(x)
            return x

CNN=Model()
#print(CNN)
losswts=torch.load('lossweights.pt')
#print(losswts)
lossfn= nn.CrossEntropyLoss(weight=losswts)
optimizer= optim.Adam(CNN.parameters(), lr=1e-5)

def train(epochs):
    for i in range(epochs):
        for batchno,(x,y) in enumerate(trainload):
            optimizer.zero_grad()
            result=CNN(x)
            loss=lossfn(result,y)
            loss.backward()
            optimizer.step()
            print(i,batchno,loss)
    print('Fin')
#epochs=5
#training=train(epochs)
