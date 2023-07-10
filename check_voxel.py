import numpy as np
import torch

test = np.load("./yugen.npy")
test = torch.tensor(test)
test = test.float()
print(test.size())
targets = []
for i in range(10):
  targets.append(1)
  targets.append(0)
# PyTorchのクラス内ではtorch.float型が前提
targets = torch.tensor(targets).float()
print(targets.dtype)

# # Trick to accept different input shapes
# x = torch.rand((1, 1) + (32,32,32))
# first_fc_in_features = 1
# print(x.size())
# for n in x.size()[1:]:
#     first_fc_in_features *= n
# x = x.view(x.size(0), -1)
# print(x.size())