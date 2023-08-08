# coding: UTF-8
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from natsort import natsorted # 普通のsortで文字列900,1000をsortすると，1000,900となってしまうため，natsortを使う

all_voxels = "./voxel_data"

for voxel in natsorted(os.listdir(all_voxels)):
    voxel_data = np.load(f"{all_voxels}/{voxel}.npy")

    # ボクセルデータの座標を作成する
    x = np.arange(0, 512)
    y = np.arange(0, 512)
    z = np.arange(0, voxel_data.shape[2])

    # RegularGridInterpolatorを作成する(元となる三次元データを作成)
    interpolator = RegularGridInterpolator((x, y, z), voxel_data ,method="cubic")

    # 30×30×30の新しいボクセルデータの座標を作成する
    new_x = np.linspace(0, 511, 30)
    new_y = np.linspace(0, 511, 30)
    new_z = np.linspace(0, voxel_data.shape[2] - 1, 30)

    # 新しいボクセルデータの座標をメッシュグリッドの座標に変換する
    xg, yg, zg = np.meshgrid(new_x, new_y, new_z, indexing='ij')

    # RegularGridInterpolatorを使用して新しいボクセルデータを補間する
    resized_data = interpolator((xg, yg, zg))

    # 0パディングされた32*32*32のボクセルデータを作る
    voxel_data_32 = np.zeros((32, 32, 32, 1), dtype=np.uint16)

    for z in range(30):
        for x in range(30):
            for y in range(30):
                voxel_data_32[x + 1][y + 1][z + 1] = resized_data[x][y][z]

    # conv3dで使いやすくするために，ボクセルデータを正規化,転置する
    voxel_data_32.T
    np.save(f"./voxel_data_32/{voxel}.npy", voxel_data_32)
    print(voxel)