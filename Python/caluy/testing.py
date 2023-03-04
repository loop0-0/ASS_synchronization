import cv2 as cv
import numpy as np
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



print(torch.__version__)               # pytorch version
print(torch.version.cuda)              # cuda 
print(torch.backends.cudnn.version())
print(torch.__version__)
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
print(my_tensor)
torch.cuda.is_available()