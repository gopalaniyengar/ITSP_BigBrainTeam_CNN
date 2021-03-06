import numpy as np
import torch
import torch.nn
"""
a=np.array([[1,2,3],[2,3,4]])
b=torch.tensor([[1,2,3,4],[2,3,4,5,]])
print(a)
print(b)

c =np.random.randn(1,2)
d=np.random.random((3,3))
e=np.random.rand(3,5)
print(c)
print(d)
#print(e)
f=c.T.reshape(1,2)
print(f)
print(np.linalg.det(d))
print(np.linalg.matrix_rank(d))
print(d.diagonal(offset=-1))
print(d.trace())
p,q=np.linalg.eig(d)
print(p)
print(q)
a=np.random.rand(2,2)
b=np.random.rand(2,2)
print(a@b)
c=np.dot(a,b)
print(c)
print(np.add(a,b))
print(np.subtract(a,b))
print(np.multiply(a,b))
print(np.linalg.inv(c))

nfilt = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
nlayer = list(nfilt.size())
print(nlayer[0])

nfilt=torch.tensor([1,1])
nconvlayer = len(list(nfilt))
print(nconvlayer)
print(torch.tensor(62))
"""

import torch
print(torch.cuda.is_available())
a=torch.tensor([1,2,3,4,5,6,7,8,9])
c=torch.tensor([1,2,3,4,5,6,7,8,9])
torch.save(a,'file.pt')
torch.save(c,'file.pt')
b,d=torch.load('file.pt')
print(b)
print(d)

