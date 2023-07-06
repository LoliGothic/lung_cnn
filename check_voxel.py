import os
import numpy as np
import torch
import pydicom
from natsort import natsorted # 普通のsortで文字列900,1000をsortすると，1000,900となってしまうため，natsortを使う

# test = np.load("./voxel_data979/1715.npy")
# test = torch.tensor(test)
# test = test.float()
# print(test.shape)

# targets = []
# for i in range(10):
#   targets.append(1)
#   targets.append(0)
# # PyTorchのクラス内ではtorch.float型が前提
# targets = torch.tensor(targets).float()
# print(targets.dtype)

# # 300番台~1700番台までのスライスされたdicom画像が保存されているディレクトリのパス
# all_patient_path = "./data"

# # 300番台の人,400番台の人...1700番台の人とループを回す
# for divide_patient_path in natsorted(os.listdir(all_patient_path)):

#     if divide_patient_path == "300-":
#         # 300番,301番...1798番,1799番とループを回す
#         for patient_path in natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}")):
#             # 300番の1枚目のdicom画像,300番の2枚目のdicom画像...1799番のhoge枚目のdicom画像とループを回す
#             for filename in natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}/{patient_path}")):
#                 dicom = pydicom.dcmread(f"{all_patient_path}/{divide_patient_path}/{patient_path}/{filename}")

#                 if dicom.pixel_array.shape != (512, 512):
#                     print(patient_path)
#                     print(filename)
#                     print(dicom.pixel_array.shape)
            
#             print(patient_path)

data = "./voxel_data"

for d in natsorted(os.listdir(data)):
    test = np.load(f"{data}/{d}")
    if test.max() != 1.0:
        print(test.max())
        print(d)
    if test.min() != 0:
        print(test.min())
        print(d)