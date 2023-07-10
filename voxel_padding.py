import numpy as np

voxel_data = np.load()

padding_voxel_data = np.zeros((32, 32, 32, 1), dtype=np.uint8)

for z in range(30):
    for x in range(30):
        for y in range(30):
            padding_voxel_data[x + 1][y + 1][z + 1] = voxel_data[x][y][z]

padding_voxel_data.T

np.save("./people_32.npy")