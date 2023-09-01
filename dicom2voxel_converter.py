import os
import pydicom
import numpy as np

# スライスされたdicom画像が保存されたフォルダのパス
slice_folder_path = "./dicom/people1/"

# スライスされたdicom画像をすべて読み込む
slice_dicoms = []
for filename in sorted(os.listdir(slice_folder_path)):
    dicom = pydicom.dcmread(os.path.join(slice_folder_path, filename))
    slice_dicoms.append(dicom)

# ボクセルデータを作成する
depth = len(slice_dicoms)
# width,height,depth,channelの配列を作る
voxels = np.zeros((512, 512, depth, 1), dtype=np.uint32)

# 各スライス画像を処理し、ボクセルデータに変換する
for z, slice_dicom in enumerate(slice_dicoms):
    slice_dicom_pixel = slice_dicom.pixel_array
    for x in range(512):
        for y in range(512):
            voxels[x][y][z] = slice_dicom_pixel[x][y]

print(voxels.max())
# conv3dで使いやすくするために，ボクセルデータを正規化,転置する
voxels = voxels / voxels.max()
voxels = voxels.T
print(voxels)
# ボクセルデータをnpyファイルとして保存する
np.save("./people1_200.npy", voxels)