import os
import time
import torch
from PIL import Image
import numpy as np
from torch import nn, optim
# torch.cat((x,y), 0)
x = torch.randn(1,5,5)
x = x.unsqueeze(0)
y = torch.randn(1,5,5)
y = y.unsqueeze(0)
z=torch.cat((x,y),0)
print(x)
print(y)
print(z)
# print(y)
# print(z)