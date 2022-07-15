import os
import cv2
import glob
import collections

import numpy as np
from skimage.metrics import structural_similarity


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def eval_score(gt_img_path, pred_img_path):
    img_gt = cv2.imread(gt_img_path).astype(np.float32)/255.
    img_pred = cv2.imread(pred_img_path).astype(np.float32)/255.

    psnr = psnr_metric(img_pred, img_gt)
    ssim = structural_similarity(img_pred, img_gt, multichannel=True)
    return dict(psnr=psnr, ssim=ssim)


def main():
    gt_files = sorted(glob.glob(os.path.join(args.src_dir, '*', 'gt', '*')))
    pred_files = list(map(lambda x: x.replace(f'{os.path.sep}gt{os.path.sep}', f'{os.path.sep}pred{os.path.sep}'), gt_files))
    pred_files = list(map(lambda x: x.replace(f'_gt.png', f'.png'), pred_files))
    print('#files=', len(gt_files))

    scores = collections.defaultdict(list)
    for gt_file, pred_file in zip(gt_files, pred_files):
        score = eval_score(gt_file, pred_file)
        for key, val in score.items():
            scores[key].append(val)

    for key, val_list in scores.items():
        print(f'{key}:\t', float(np.mean(val_list)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, 
        default='./EXPERIMENTS/zju/images_v3'
    )
    args = parser.parse_args()

    main()