import os
import pydicom
import numpy as np
from natsort import natsorted # 普通のsortで文字列900,1000をsortすると，1000,900となってしまうため，natsortを使う

# 300番台~1700番台までのスライスされたdicom画像が保存されているディレクトリのパス
all_patient_path = "./data"

# 300番台の人,400番台の人...1700番台の人とループを回す
for divide_patient_path in natsorted(os.listdir(all_patient_path)):
        # 300番,301番...1798番,1799番とループを回す
        for patient_path in natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}")):
            # 患者１人ずつのボクセルデータを作りたいため，patient_pathが変わるたびに，slice_dicomsを空にする
            slice_dicoms = []

                # 300番の1枚目のdicom画像,300番の2枚目のdicom画像...1799番のhoge枚目のdicom画像とループを回す
                for filename in natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}/{patient_path}")):
                    dicom = pydicom.dcmread(f"{all_patient_path}/{divide_patient_path}/{patient_path}/{filename}")
                    slice_dicoms.append(dicom)
                    
                    print(filename)
                
                # ボクセルデータを作成する
                depth = len(slice_dicoms)
                # width,height,depth,channelの配列を作る
                voxels = np.zeros((512, 512, depth, 1), dtype=np.uint8)

                # 各スライス画像を処理し、ボクセルデータに変換する
                for z, slice_dicom in enumerate(slice_dicoms):
                    slice_dicom_pixel = slice_dicom.pixel_array
                    for x in range(512):
                        for y in range(512):
                            voxels[x][y][z] = slice_dicom_pixel[x][y]

                # conv3dで使いやすくするために，ボクセルデータを正規化,転置する
                voxels = voxels / 255
                voxels = voxels.T

                # ボクセルデータをnpyファイルとして保存する
                np.save(f"./voxel_data/{patient_path}.npy", voxels)
                print(patient_path)