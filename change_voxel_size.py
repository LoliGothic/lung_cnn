import numpy as np
from scipy.interpolate import RegularGridInterpolator

voxel_data = np.load("./people1_200.npy")

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
print(interpolator.method)

# リサイズされたボクセルデータのサイズを確認する
print(resized_data.shape)