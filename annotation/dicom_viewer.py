import os
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import math
import clipping
from natsort import natsorted
from tkinter import filedialog
from matplotlib.widgets import Slider

while(True):
    init_dir = "../"
    slice_folder_path = filedialog.askdirectory(initialdir = init_dir)  #DICOMファイルのフォルダを入力で得る
    file_list = []          #フォルダ内すべてのファイル
    dicom_list = []         #フォルダ内のDICOMファイル
    slice_parameter = []    #DICOMファイルのSliceLocationの値

    #フォルダ内のDICOMデータをリストにして取得する関数。DICOMのSliceLocationタグの値で並びかえる
    def get_dicom_file_list(slice_folder_path):
        """Return the dicom file list sorted by SliceLocation
        Args
            arg(str): slice_folder_path of the dicom_files

        Returns
            return(list): dicom_file list sorted by SliceLocation

        Note
            Check if the dicom file has .SliceLocation attribute
        """

        for filename in natsorted(os.listdir(slice_folder_path)):
            file_absname = os.path.join(slice_folder_path, filename)
            ds = pydicom.read_file(file_absname)
            slice_parameter.append(ds.SliceLocation)
            dicom_list.append(file_absname)

        df = pd.DataFrame(zip(dicom_list, slice_parameter), columns=['filename', 'slicelocation'])
        df_sorted = df.sort_values('slicelocation')
        sorted_dicom_list = list(df_sorted['filename'])
        return sorted_dicom_list

    # 並べ替え。慣れてる人はラムダ関数のほうが短く書けます。
    sorted_dicom_list = get_dicom_file_list(slice_folder_path)
    file_number = len(sorted_dicom_list)
    init_slice = int(file_number / 2)   #真ん中のスライスを初期値に設定
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [20, 1]})
    axes = {ax0:'ax0', ax1:'ax1'}
    # ２行からなるfigを定義。ax0がDICOM画像の表示、ax1がスライダーの位置

    # DICOMデータからピクセルデータを得る関数を定義
    def get_pixel(dcm_index):
        """get the pixel_array from DICOM file
            arg(int): index of the dicom_file in the dicom_list
            return(ndarray) : pixel_array of the dicom_file
        """
        ds = pydicom.read_file(sorted_dicom_list[dcm_index])
        return ds.pixel_array

    # 初期の画像を表示する
    image_table = ax0.imshow(get_pixel(init_slice), cmap='gray', vmax = 1500, vmin = -500)

    # Sliderの属性を定義する
    slice_slider = Slider(
        ax=ax1,                     #subplotのax1にスライダーを設置
        label="Slice Number",       #「Slice Number」というタイトル
        valinit=init_slice,         # 初期値の設定　ここでは真ん中
        valmin=0,                   # 最小値の設定
        valmax=len(dicom_list),     # 最大値の設定（ファイル数 - 1)
        valfmt='%d',                # int型
        orientation="horizontal"    # 水平方向にスライドする
    )

    # Sliderの値が変わった時に行う動作
    def update(val):
        """Set the valuable to the slider value"""
        image_table.set_data(get_pixel(int(slice_slider.val)))
        fig.canvas.draw_idle()      # pltのfigをリニューアルする
        return

    #Sliderの動きに合わせてアップデートする
    slice_slider.on_changed(update)

    def onclick(event):
        if axes[event.inaxes] != "ax0" or event.button != 1 or event.xdata == None or event.ydata == None:
            return
        else:
            print('{} click: button={}, x={}, y={}, xdata={}, ydata={}'.format(
                'double' if event.dblclick else 'single', event.button,
                event.x, event.y, event.xdata, event.ydata,
            ))
            clipping.clipping(round(event.ydata), round(event.xdata), math.floor(slice_slider.val), slice_folder_path)

            # viewerを閉じる
            plt.cla()
            plt.clf()
            plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()