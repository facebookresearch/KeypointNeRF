import os
import copy
import random

import cv2
import torch
import imageio
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from kornia.geometry.conversions import convert_points_to_homogeneous 


def get_human_info(split, test_mode):
    if split == 'train':
        human_info = {
            'CoreView_313': {'begin_i': 0, 'i_intv': 1, 'ni': 60},
            'CoreView_315': {'begin_i': 0, 'i_intv': 6, 'ni': 400},
            'CoreView_377': {'begin_i': 0, 'i_intv': 30, 'ni': 300},
            'CoreView_386': {'begin_i': 0, 'i_intv': 6, 'ni': 300},
            'CoreView_390': {'begin_i': 700, 'i_intv': 6, 'ni': 300},
            'CoreView_392': {'begin_i': 0, 'i_intv': 6, 'ni': 300},
            'CoreView_396': {'begin_i': 810, 'i_intv': 5, 'ni': 270}
        }

    elif split == 'test' or split == 'val':
        if test_mode == 'model_o_motion_o':
            human_info = {
                'CoreView_313': {'begin_i': 0, 'i_intv': 1, 'ni': 60},
                'CoreView_315': {'begin_i': 0, 'i_intv': 1,
                                 'ni': 400},
                'CoreView_377': {'begin_i': 0, 'i_intv': 1,
                                 'ni': 300},
                'CoreView_386': {'begin_i': 0, 'i_intv': 1,
                                 'ni': 300},
                'CoreView_390': {'begin_i': 700, 'i_intv': 1,
                                 'ni': 300},
                'CoreView_392': {'begin_i': 0, 'i_intv': 1,
                                 'ni': 300},
                'CoreView_396': {'begin_i': 810, 'i_intv': 1,
                                 'ni': 270}}


        elif test_mode == 'model_o_motion_x':
            human_info = {
                'CoreView_313': {'begin_i': 60, 'i_intv': 1, 'ni': 1000},
                'CoreView_315': {'begin_i': 400, 'i_intv': 1, 'ni': 1000},
                'CoreView_377': {'begin_i': 300, 'i_intv': 1, 'ni': 317},
                'CoreView_386': {'begin_i': 300, 'i_intv': 1, 'ni': 346},
                'CoreView_390': {'begin_i': 0, 'i_intv': 1, 'ni': 700},
                'CoreView_392': {'begin_i': 300, 'i_intv': 1, 'ni': 256},
                'CoreView_396': {'begin_i': 1080, 'i_intv': 1, 'ni': 270}}

        elif test_mode == 'model_x_motion_x':

            human_info = {
                'CoreView_387': {'begin_i': 0, 'i_intv': 1, 'ni': 654},
                'CoreView_393': {'begin_i': 0, 'i_intv': 1, 'ni': 658},
                'CoreView_394': {'begin_i': 0, 'i_intv': 1, 'ni': 859},
            }

    return human_info

class ZJUDataset(data.Dataset):
    def __init__(self, data_root, split, **kwargs):
        super(ZJUDataset, self).__init__()

        zju_313_315_sample_cam = [3, 5, 10, 12, 18, 21]
        zju_sample_cam = [3, 5, 10, 12, 18, 20]
        self.zju_313_315_sample_cam = zju_313_315_sample_cam
        self.zju_sample_cam = zju_sample_cam
        self.test_input_view = [0,7,15]
        self.data_root = data_root
        self.max_len = kwargs.get('max_len', -1)

        self.split = split  # 'train'
        test_mode = kwargs.get('test_mode', 'model_x_motion_x')
        self.test_mode = test_mode
        run_mode = split
        self.im2tensor = self.image_to_tensor()
        self.ratio = 0.5
        self.jitter = None
        if kwargs.get('use_color_jitter', False):
            self.jitter = self.color_jitter()

        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.start_end = {}

        human_info = get_human_info(self.split, test_mode)
        human_list = list(human_info.keys())

        if self.split in ['test', 'val']:
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        for idx in range(len(human_list)):
            human = human_list[idx]

            data_root = os.path.join(data_root, human)
            ann_file = os.path.join(self.data_root, human, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()

            self.cams[human] = annots['cams']

            num_cams = len(self.cams[human]['K'])

            if run_mode == 'train':
                test_view = [i for i in range(num_cams)]

            elif run_mode == 'test' or run_mode == 'val':
                if human in ['CoreView_313', 'CoreView_315']:
                    test_view = zju_313_315_sample_cam
                else:
                    test_view = zju_sample_cam

            if len(test_view) == 0:
                test_view = [0]

            i = 0
            i = i + human_info[human]['begin_i']
            i_intv = human_info[human]['i_intv']
            ni = human_info[human]['ni']

            ims = np.array([
                np.array(ims_data['ims'])[test_view]
                for ims_data in annots['ims'][i:i + ni][::i_intv]
            ]).ravel()

            cam_inds = np.array([
                np.arange(len(ims_data['ims']))[test_view]
                for ims_data in annots['ims'][i:i + ni][::i_intv]
            ]).ravel()


            start_idx = len(self.ims)
            length = len(ims)
            self.ims.extend(ims)
            self.cam_inds.extend(cam_inds)

            if human in ['CoreView_313', 'CoreView_315']:

                self.ims[start_idx:start_idx + length] = [
                    data_root + '/' + x.split('/')[0] + '/' +
                    x.split('/')[1].split('_')[4] + '.jpg' for x in
                    self.ims[start_idx:start_idx + length]]
            else:
                self.ims[start_idx:start_idx + length] = [
                    data_root + '/' + x for x in
                    self.ims[start_idx:start_idx + length]]

            self.start_end[human] = {}
            self.start_end[human]['start'] = int(self.ims[start_idx].split('/')[-1][:-4])
            self.start_end[human]['end'] = int(self.ims[start_idx + length - 1].split('/')[-1][:-4])
            self.start_end[human]['length'] = self.start_end[human]['end'] - self.start_end[human]['start']
            self.start_end[human]['intv'] = human_info[human]['i_intv']

        # self.nrays = cfg.N_rand
        self.num_humans = len(human_list)

    @classmethod
    def from_config(cls, dataset_cfg, data_split, cfg):
        ''' Creates an instance of the dataset.

        Args:
            dataset_cfg (dict): input configuration.
            data_split (str): data split (`train` or `val`).
        '''
        assert data_split in ['train', 'val', 'test']

        dataset_cfg = copy.deepcopy(dataset_cfg)
        dataset_cfg['is_train'] = data_split == 'train'
        if f'{data_split}_cfg' in dataset_cfg:
            dataset_cfg.update(dataset_cfg[f'{data_split}_cfg'])
        if dataset_cfg['is_train']:
            dataset = cls(split=data_split, **dataset_cfg)
        else:
            dataset = ZJUTestDataset(split=data_split, **dataset_cfg)
        return dataset
    
    def color_jitter(self):
        ops = []

        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )

        return transforms.Compose(ops)
    
    def image_to_tensor(self):

        ops = []

        ops.extend(
            [transforms.ToTensor(), ]
        )

        return transforms.Compose(ops)

    def get_mask(self, index):

        data_info = self.ims[index].split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        msk_exist = False
        msk_cihp_exist = False

        msk_path = os.path.join(self.data_root, human, 'mask', camera, frame)[:-4] + '.png'
        msk_exist = os.path.exists(msk_path)
        if msk_exist:
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

        msk_path = os.path.join(self.data_root, human, 'mask_cihp', camera, frame)[:-4] + '.png'
        msk_cihp_exist = os.path.exists(msk_path)
        if msk_cihp_exist:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)
        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)
        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        border = 5

        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def get_input_mask(self, human, index, filename):
        # index is camera index

        msk_exist = False
        msk_cihp_exist = False

        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(self.data_root, human, 'mask', 'Camera (' + str(index) + ')', filename[:-4] + '.png')
        else:
            msk_path = os.path.join(self.data_root, human, 'mask', 'Camera_B' + str(index), filename[:-4] + '.png')

        msk_exist = os.path.exists(msk_path)

        if msk_exist:
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(self.data_root, human, 'mask_cihp', 'Camera (' + str(index) + ')', filename[:-4] + '.png')
        else:
            msk_path = os.path.join(self.data_root, human, 'mask_cihp', 'Camera_B' + str(index), filename[:-4] + '.png')
        msk_cihp_exist = os.path.exists(msk_path)

        if msk_cihp_exist:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)

        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)

        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        return msk
    
    # def get_smpl_vertice(self, human, frame):
    #     vertices_path = os.path.join(self.data_root, human, 'vertices', '{}.npy'.format(frame))
    #     smpl_vertice = np.load(vertices_path).astype(np.float32)
    #     return smpl_vertice

    def get_smpl_joints(self, human, frame):
        vertices_path = os.path.join(self.data_root, human, 'joints3d', '{}.npy'.format(frame))
        smpl_vertice = np.load(vertices_path).astype(np.float32)
        return smpl_vertice

    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, tar_index):
        # sample a frame for training
        tar_img_path = self.ims[tar_index]
        tar_data_info = tar_img_path.split('/')
        human = tar_data_info[-3]
        frame = tar_data_info[-1]
        target_frame = frame[:-4]
        frame_index = int(target_frame)
        zfill = len(target_frame)

        current_frame = int(target_frame)
        filename = str(current_frame).zfill(zfill) + '.jpg'

        input_view = copy.deepcopy(self.test_input_view)
        if human in ['CoreView_313', 'CoreView_315']:
            all_input_view = [i for i in range(len(self.cams[human]['K']))]
            cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22]
        else:
            all_input_view = [i for i in range(len(self.cams[human]['K']))]

        if self.split == 'train':
            tar_pool = list(set(all_input_view) - set(input_view))
            random.shuffle(tar_pool)
            tar_view_ind = tar_pool[0]
        else:
            tar_view_ind = self.cam_inds[tar_index]

        kernel = np.ones((5, 5), np.uint8)
        input_view = [tar_view_ind] + input_view
        input_imgs, input_msks, input_K, input_Rt = [], [], [], []
        # input_msks_dial = []
        for idx in input_view:
            # prep image path and corresponding mask
            if human in ['CoreView_313', 'CoreView_315']:
                cam_idx = cam_idx_list[idx]
                input_img_path = os.path.join(self.data_root, human, 'Camera (' + str(cam_idx + 1) + ')', filename)
                input_msk = self.get_input_mask(human, cam_idx + 1, filename)
            else:
                input_img_path = os.path.join(self.data_root, human, 'Camera_B' + str(idx + 1), filename)
                input_msk = self.get_input_mask(human, idx + 1, filename)

            # load data
            in_K, in_D = np.array(self.cams[human]['K'][idx]).astype(np.float32), np.array(self.cams[human]['D'][idx]).astype(np.float32)
            in_R, in_T = np.array(self.cams[human]['R'][idx]).astype(np.float32), (np.array(self.cams[human]['T'][idx]) / 1000.).astype(np.float32)
            in_Rt = np.concatenate((in_R.reshape((3,3)), in_T.reshape(3, 1)), axis=1)
            input_img = imageio.imread(input_img_path).astype(np.float32) / 255.
            input_img, input_msk = cv2.undistort(input_img, in_K, in_D), cv2.undistort(input_msk, in_K, in_D)

            # resize images
            H, W = int(input_img.shape[0] * self.ratio), int(input_img.shape[1] * self.ratio)
            input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)

            # apply foreground mask
            input_img[input_msk == 0] = 0
            input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)

            # apply foreground mask
            border = 5
            kernel = np.ones((border, border), np.uint8)
            input_msk = cv2.erode(input_msk.astype(np.uint8) * 255, kernel)

            # [0,1]
            # input_msks_dial.append(self.im2tensor(cv2.dilate(input_msk.copy(), kernel) > 0.0))
            input_img, input_msk = self.im2tensor(input_img), self.im2tensor(input_msk).bool()
            in_K[:2] = in_K[:2] * self.ratio

            # append data
            input_imgs.append(input_img)
            input_msks.append(input_msk)
            input_K.append(torch.from_numpy(in_K))
            input_Rt.append(torch.from_numpy(in_Rt))

        i = int(frame[:-4])
        joints_path = os.path.join(self.data_root, human, 'joints3d', f'{i}.npy')
        xyz_joints = np.load(joints_path).astype(np.float32)
        smpl_joints = np.array(xyz_joints).astype(np.float32)
        human_idx = 0
        if self.split in ['test', 'val']:
            human_idx = self.human_idx_name[human]
        
        ret = {
            'images': torch.stack(input_imgs),
            'images_masks': torch.stack(input_msks),
            # 'images_masks_dial': torch.stack(input_msks_dial),
            'K': torch.stack(input_K),
            'Rt': torch.stack(input_Rt),
            'kpt3d': torch.from_numpy(smpl_joints),
            'i': i,
            'human_idx': human_idx,
            'sessision': human,
            'frame_index': frame_index,
            'human': human,
            'cam_ind': input_view[0],
            "index": {"camera": "cam", "segment": 'zju', "tar_cam_id": tar_view_ind,
                "frame": f"{human}_{frame_index}", "ds_idx": idx},
        }
        if self.split in ['test', 'val']:  # add root pose for rendering
            smpl_root_orient = np.load(os.path.join(self.data_root, human, 'params', f'{i}.npy'), allow_pickle=True).item()['Rh']
            smpl_root_orient = smpl_root_orient.reshape(-1)
            smpl_root_orient, _ = cv2.Rodrigues(smpl_root_orient)
            smpl_root_orient = torch.from_numpy(smpl_root_orient)

            headpose = torch.eye(4)
            headpose[:3, :3] = input_Rt[1][:3, :3].t()
            headpose[:3, :3] = smpl_root_orient
            # _dr = np.zeros((3,), dtype=np.float32)
            # _dr[0] = np.pi
            # _dr[1] = -np.pi
            # _dr, _ = cv2.Rodrigues(_dr)
            # _dr, _ = cv2.Rodrigues(smpl_root_orient)
            # headpose[:3, :3] = torch.from_numpy(_dr)
            # headpose[:3, 0] = torch.tensor([1,0,0])
            # headpose[:3, 1] = -headpose[:3, 1]  # flip y axis
            headpose[:3, 3] = torch.from_numpy(smpl_joints[0])
            ret['headpose'] = headpose

        if self.jitter is not None:
            ret['images'] = self.jitter(ret['images'])*ret['images_masks']
        # if self.split in ['test', 'val']:
        bounds = self.load_human_bounds(human, i)
        ret['mask_at_box'] = self.get_mask_at_box(
            bounds,
            input_K[0].numpy(),
            input_Rt[0][:3, :3].numpy(),
            input_Rt[0][:3, -1].numpy(),
            H, W)
        ret['bounds'] = bounds
        ret['mask_at_box'] = ret['mask_at_box'].reshape((H, W))
        if False:  # TODO TMP: overly input images with keypoints for the teaser
            _kpts3d = convert_points_to_homogeneous(ret['kpt3d']).t()
            _kpts2d = [((ret['K'][i] @ ret['Rt'][i]) @ _kpts3d).t() for i in range(ret['K'].shape[0])]
            _kpts2d = [kp2[:, :2]/kp2[:, 2, None] for kp2 in _kpts2d]

            new_imgs = []  # V, 3, H, W
            for i in range(ret['K'].shape[0]):
                img = ret['images'][i].permute(1, 2, 0).numpy().copy()  # V, H, W, 3
                img = (img*255.).astype(np.uint8)
                img = draw_keypoints(img, _kpts2d[i])
                img = (img.astype(np.float32)/255.)
                new_imgs.append(torch.from_numpy(img))
            new_imgs = torch.stack(new_imgs)
            ret['images'] = new_imgs.permute(0, 3, 1, 2) # VHW3 -> V3HW

        return ret

    def get_length(self):
        return self.__len__()

    def __len__(self):
        if self.max_len == -1:
            return len(self.ims)
        else:
            return min(len(self.ims), self.max_len)

    def load_human_bounds(self, human, i):
        vertices_path = os.path.join(self.data_root, human, 'vertices', f'{i}.npy')
        xyz = np.load(vertices_path).astype(np.float32)
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds
        

    @staticmethod
    def get_mask_at_box(bounds, K, R, T, H, W):
        ray_o, ray_d = ZJUDataset.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = ZJUDataset.get_near_far(bounds, ray_o, ray_d)
        return mask_at_box.reshape((H, W))


    @staticmethod
    def get_rays(H, W, K, R, T):
        rays_o = -np.dot(R.T, T).ravel()

        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32), indexing='xy')

        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        rays_d = pixel_world - rays_o[None, None]
        rays_o = np.broadcast_to(rays_o, rays_d.shape)

        return rays_o, rays_d

    @staticmethod
    def get_near_far(bounds, ray_o, ray_d, boffset=(-0.01, 0.01)):
        """calculate intersections with 3d bounding box"""
        bounds = bounds + np.array([boffset[0], boffset[1]])[:, None]  # TODO consider increasing this for my training
        nominator = bounds[None] - ray_o[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        ray_d[np.abs(ray_d) < 1e-5] = 1e-5
        d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
        # calculate the six interections
        p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
        # calculate the intersections located at the 3d bounding box
        min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
        eps = 1e-6
        p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                        (p_intersect[..., 0] <= (max_x + eps)) * \
                        (p_intersect[..., 1] >= (min_y - eps)) * \
                        (p_intersect[..., 1] <= (max_y + eps)) * \
                        (p_intersect[..., 2] >= (min_z - eps)) * \
                        (p_intersect[..., 2] <= (max_z + eps))
        # obtain the intersections of rays which intersect exactly twice
        mask_at_box = p_mask_at_box.sum(-1) == 2
        p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
            -1, 2, 3)

        # calculate the step of intersections
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
        d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)

        return near, far, mask_at_box

def draw_keypoints(img, kpts, color=(255, 0, 0), size=3):
    for i in range(kpts.shape[0]):
        kp2 = kpts[i].tolist()
        kp2 = [int(kp2[0]), int(kp2[1])]
        img = cv2.circle(img, kp2, 2, color, size)
    return img

class ZJUTestDataset(ZJUDataset):
    def __init__(self, data_root, split, **kwargs):
        super().__init__(data_root, split, **kwargs)

        # load im list
        self.sc_factor = 1.0
        human_info = get_human_info(self.split, self.test_mode)
        human_list = list(human_info.keys())
        _inds = []
        inds = np.arange(0, len(self.ims))

        start = 0

        for human_idx in range(len(human_list)):
            human = human_list[human_idx]

            ni = human_info[human]['ni']

            if human in ['CoreView_313', 'CoreView_315']:
                num_cams = len(self.zju_313_315_sample_cam)
            else:
                num_cams = len(self.zju_sample_cam)

            num_frames = ni

            sub_len = num_frames * num_cams
            sub_inds = inds[start:start + sub_len]
            sub_inds = sub_inds.reshape(num_frames, -1)[::30]

            _inds.extend(sub_inds.ravel())
            start = start + sub_len

        self.ims = [self.ims[_i] for _i in _inds]
        self.cam_inds = [self.cam_inds[_i] for _i in _inds]
        filter_list = [
            # 'CoreView_387/Camera_B6/000330.jpg', 
            # 'CoreView_393/Camera_B6/000060.jpg',
            # 'CoreView_393/Camera_B6/000120.jpg',
            # 'CoreView_394/Camera_B6/000150.jpg',
            # 'CoreView_394/Camera_B6/000540.jpg',
            # 'CoreView_394/Camera_B6/000750.jpg',
        ]
        if filter_list != []:
            new_inds = []
            for _ind, im in enumerate(self.ims):
                if any([f in im for f in filter_list]):
                    new_inds.append(_ind)
            self.ims = [self.ims[_i] for _i in new_inds]
            self.cam_inds = [self.cam_inds[_i] for _i in new_inds]
