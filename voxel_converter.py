from PIL import Image
import numpy as np
import os

# スライスされたJPEG画像が保存されたフォルダのパス
slice_folder_path = "path/to/slice/folder"

# 1つのスライス画像のサイズ
width, height = 512, 512

# スライスされたJPEG画像をすべて読み込む
slice_images = []
for filename in sorted(os.listdir(slice_folder_path)):
    if filename.endswith(".jpg"):
        slice_image = Image.open(os.path.join(slice_folder_path, filename))
        slice_images.append(slice_image)

# ボクセルデータを作成する
depth = len(slice_images)
voxels = np.zeros((width, height, depth, 3), dtype=np.uint8)

# 各スライス画像を処理し、ボクセルデータに変換する
for z, slice_image in enumerate(slice_images):
    for x in range(width):
        for y in range(height):
            r, g, b = slice_image.getpixel((x, y))
            voxels[x][y][z] = [r, g, b]

# ボクセルデータをnpyファイルとして保存する
np.save("path/to/voxels.npy", voxels)