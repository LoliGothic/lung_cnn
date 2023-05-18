import numpy as np
import torch

test = np.load("./yugen.npy")
test = torch.tensor(test)
test = test.float()

targets = []
for i in range(10):
  targets.append(1)
  targets.append(0)
# PyTorchのクラス内ではtorch.float型が前提
targets = torch.tensor(targets).float()
print(targets.dtype)