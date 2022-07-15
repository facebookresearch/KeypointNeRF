import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

        self.result_dir = None

    @staticmethod
    def psnr_metric(img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index):
        # mask_at_box = batch['mask_at_box'].squeeze()
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = rgb_pred[y:y + h, x:x + w]
        img_gt = rgb_gt[y:y + h, x:x + w]

        human_dir = os.path.join(self.result_dir, human_idx)
        pred_dir = os.path.join(human_dir, 'pred')
        gt_dir = os.path.join(human_dir, 'gt')
        input_dir = os.path.join(human_dir, 'input')

        for _p in [pred_dir, gt_dir, input_dir]:
            os.system(f'mkdir -p {_p}')

        cv2.imwrite(os.path.join(pred_dir, f'frame{frame_index}_view{view_index}.png'), (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(os.path.join(gt_dir, f'frame{frame_index}_view{view_index}_gt.png'), (img_gt[..., [2, 1, 0]] * 255))
        # print(os.path.join(pred_dir, f'frame{frame_index}_view{view_index}.png'))

        input_imgs = (input_imgs[..., [2, 1, 0]] * 255.).astype(np.uint8)
        for view in range(input_imgs.shape[0]):
            cv2.imwrite(os.path.join(input_dir, f'frame{frame_index}_t_0_view_{view}.png'), input_imgs[view])

        # compute the ssim
        ssim = structural_similarity(img_pred, img_gt, multichannel=True)
        return ssim

    def compute_score(self, rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index):
        """
        Args:
            rgb_pred (torch.tensor): (1, 3, H, W) [0,1]
            rgb_gt (torch.tensor): (1, 3, H, W) [0,1]
            mask_at_box (torch.tensor): (1, H, W)
            input_imgs (torch.tensor): (V, 3, H, W) [0,1]

            human_idx (str): id of the human 
            human_idx (): 
            frame_index (): 
            view_index (): 
        """
        rgb_pred = rgb_pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        input_imgs = input_imgs.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()  # V, H, W, 1
        rgb_gt = rgb_gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        mask_at_box = mask_at_box.squeeze().detach().cpu().numpy()  # H, W

        mse = np.mean((rgb_pred - rgb_gt) ** 2)
        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        ssim = self.ssim_metric(rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index)

        return {
            'mse': float(mse), 
            'psnr': float(psnr),
            'ssim': float(ssim)
        }

    # def evaluate(self, output, batch):
    #     rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
    #     rgb_gt = batch['rgb'][0].detach().cpu().numpy()
    #     mse = np.mean((rgb_pred - rgb_gt) ** 2)
    #     self.mse.append(mse)
    #     psnr = self.psnr_metric(rgb_pred, rgb_gt)
    #     self.psnr.append(psnr)
    #     ssim = self.ssim_metric(rgb_pred, rgb_gt, batch, output)
    #     self.ssim.append(ssim)
    #     mse_str = 'mse: {}'.format(np.mean(self.mse))
    #     psnr_str = 'psnr: {}'.format(np.mean(self.psnr))
    #     ssim_str = 'ssim: {}'.format(np.mean(self.ssim))
    #     print(mse_str)
    #     print(psnr_str)
    #     print(ssim_str)

    # def summarize(self):
    #     os.system(f'mkdir -p {self.result_dir}')
    #     # metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
    #     np.save(os.path.join(self.result_dir, 'mse.npy'), self.mse)
    #     np.save(os.path.join(self.result_dir, 'psnr.npy'), self.psnr)
    #     np.save(os.path.join(self.result_dir, 'ssim.npy'), self.ssim)
    #     mse_str = 'mse: {}'.format(np.mean(self.mse))
    #     psnr_str = 'psnr: {}'.format(np.mean(self.psnr))
    #     ssim_str = 'ssim: {}'.format(np.mean(self.ssim))
    #     print(f'{psnr_str}\t{ssim_str}')
    #     with open(os.path.join(self.result_dir, 'summary.txt'), 'w') as out:
    #         out.writelines([mse_str, psnr_str, ssim_str])
    #     self.mse = []
    #     self.psnr = []
    #     self.ssim = []

