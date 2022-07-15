# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse

from os.path import join

parser = argparse.ArgumentParser('Rename inconsistent image files.')
parser.add_argument('--data_dir', default='./data/zju_mocap', type=str, help='Data direcotry.')
args = parser.parse_args()

if __name__ == '__main__':
    data_dir = args.data_dir
    root_path_list = [
        join(data_dir,'CoreView_313'), join(data_dir,'CoreView_313', 'mask_cihp'), join(data_dir,'CoreView_313', 'mask'),
        join(data_dir,'CoreView_315'), join(data_dir,'CoreView_315', 'mask_cihp'), join(data_dir,'CoreView_315', 'mask'),
    ]
    cam_lists = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23]

    for root_path in root_path_list:
        for cam_idx in cam_lists:
            img_folder= join(root_path, f'Camera ({cam_idx})')
            files = [f for f in os.listdir(img_folder) if os.path.isfile(join(img_folder, f))]

            for file in files:
                if os.path.basename(root_path) in ['mask_cihp']:
                    fname = f"{file.split('_')[4]}.png"
                elif os.path.basename(root_path) in ['CoreView_313','CoreView_315']:
                    fname = f"{file.split('_')[4]}.jpg"

                os.rename(join(img_folder, file), join(img_folder, fname))
    