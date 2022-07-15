# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
import pathlib
import functools

from typing import List

import cv2
import torch
import torchvision
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as thf

def cond_mkdir(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_360cameras(headpose, focal, trans, sc_factor, im_w, im_h, znear, zfar, n_frames=90):
    device = headpose.device
    T_i = torch.eye(4, device=device)
    T_i[:3, :4] = headpose[:3, :4]
    T_i[:3, :3] = T_i[:3, :3].t()
    T_i[:3, 3] = - T_i[:3, :3] @ T_i[:3, 3]

    tar_cameras = []
    theta = 0
    for idx in range(n_frames):
        # compute target view
        dr = np.zeros((3,), dtype=np.float32)
        dr[0] = np.pi
        dR1, _ = cv2.Rodrigues(dr)
        dr = np.zeros((3,), dtype=np.float32)
        dr[1] = theta
        dR2, _ = cv2.Rodrigues(dr)
        dR = (dR1 @ dR2).astype(np.float32)

        dt = np.asarray([0, 0, trans], dtype=np.float32)
        K = np.asarray([[focal, 0, im_w / 2],
                        [0, focal, im_h / 2],
                        [0, 0, 1]], dtype=np.float32)

        extrin_tar = torch.eye(4)
        intrin_tar = torch.eye(4)
        extrin_tar[:3, :3] = torch.from_numpy(dR)
        extrin_tar[:3, 3] = torch.from_numpy(dt)
        intrin_tar[:3, :3] = torch.from_numpy(K)

        # rescale extrinsics
        extrinsic = torch.matmul(extrin_tar.to(device=device), T_i).clone()
        extrinsic[:3, 3] *= sc_factor
        # world2cams = torch.eye(4)
        # world2cams[:3, :4] = extrinsic

        intrin_tar = intrin_tar.cuda().unsqueeze(0)
        # update angle
        theta = theta + 2.0 * np.pi / n_frames

        # set rendering camera paramters
        tar_cameras.append({
            'w2cs': extrinsic.to(device=device),
            'c2ws': torch.inverse(extrinsic).to(device=device),
            'intrinsics': intrin_tar.to(device=device),
            'im_w': im_w, 'im_h': im_h,
            'znear': znear, 'zfar': zfar,
        })

    return tar_cameras

def feat_sample(feat, uv, mode='bilinear', padding_mode='border', align_corners=True):
    '''
    args:
        feat: (B, C, H, W)
        uv: (B, N, 2) [-1, 1]
    return:
        (B, N, C)
    '''
    uv = uv[:, :, None]
    feat = thf.grid_sample(
        feat,
        uv,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return feat.view(*feat.shape[:2], -1).permute(0, 2, 1)

@torch.jit.script
def fused_mean_variance(x, x_weights):
    mean = torch.sum(x * x_weights, dim=2, keepdim=True)
    var = torch.sum(x_weights * (x - mean) ** 2, dim=2, keepdim=True)
    return mean, var

def compute_error(out_nerf=None, vggloss=None, lambdas={}):
    err_dict = {}
    err_dict.update(compute_error_nerf(out_nerf, lambdas, vggloss))

    loss = 0.0
    for k, v in err_dict.items():
        loss += v
    err_dict["e_all"] = loss

    return loss, err_dict

def compute_error_nerf(out_nerf, lambdas, vggloss):
    err_dict = {}
    lambda_l1_c = lambdas.get("lambda_l1_c", 10.0)
    lambda_l1 = lambdas.get("lambda_l1", 10.0)
    lambda_l2 = lambdas.get("lambda_l2", 0.0)
    lambda_lp = lambdas.get("lambda_lp", 0.0)
    lambda_ssim = lambdas.get("lambda_ssim", 0.0)
    lambda_vgg = lambdas.get("lambda_vgg", 1.0)
    lambda_aux = lambdas.get("lambda_aux", 1.0)
    lambda_mloss = lambdas.get("lambda_mloss", 0.0)

    pix_weights = {"l1": lambda_l1,
                   "l2": lambda_l2,
                   "lp": lambda_lp,
                   "ssim": lambda_ssim}

    imp_sample = {}
    for k, v in lambdas.items():
        if "top" in k:
            imp_sample[k.replace("lambda_", "")] = v

    loss_pix_c = 0.0
    if "tex_cal" in out_nerf and lambda_l1_c > 0.0:
        losses = pix_loss(out_nerf["tex_cal"], out_nerf["tar_img"], {"l1": lambda_l1_c})
        loss_pix_c += losses["l1"]
    if "tex_aux_cal" in out_nerf and lambda_l1_c > 0.0 and lambda_aux > 0.0:
        losses = pix_loss(out_nerf["tex_aux_cal"], out_nerf["tar_img"], {"l1": lambda_l1_c})
        loss_pix_c += lambda_aux * losses["l1"]
    if loss_pix_c > 0.0:
        err_dict["e_pix_c"] = loss_pix_c

    loss_pix_fine, loss_pix_fine_aux = {}, {}
    if "tex_cal_fine" in out_nerf:
        loss_pix_fine = pix_loss(out_nerf["tex_cal_fine"], out_nerf["tar_img"], pix_weights)
        for k, v in loss_pix_fine.items():
            err_dict[f"e_pix_{k}"] = v
    if "tex_aux_cal_fine" in out_nerf and lambda_aux > 0.0:
        loss_pix_fine_aux = pix_loss(
            out_nerf["tex_aux_cal_fine"], out_nerf["tar_img"], pix_weights)
        for k, v in loss_pix_fine_aux.items():
            err_dict[f"e_pix_{k}a"] = lambda_aux * v

    if "alpha" in out_nerf and 'tar_alpha' in out_nerf and lambda_mloss > 0.0:
        err_dict["mask_loss_c"] = lambda_mloss*thf.mse_loss(
            out_nerf['alpha'].clip(1e-3, 1.0).squeeze(),
            out_nerf['tar_alpha'].squeeze())

    if "alpha_fine" in out_nerf and 'tar_alpha' in out_nerf and lambda_mloss > 0.0:
        err_dict["mask_loss_f"] = lambda_mloss*thf.mse_loss(
            out_nerf['alpha_fine'].clip(1e-3, 1.0).squeeze(),
            out_nerf['tar_alpha'].squeeze())

    if vggloss is not None:
        loss_vgg = 0.0
        if "tex_cal_fine" in out_nerf:
            nerf_tex_mskd = out_nerf["tex_cal_fine"]
            loss_vgg += lambda_vgg * vggloss(nerf_tex_mskd, out_nerf["tar_img"])
        if "tex_aux_cal_fine" in out_nerf and lambda_aux > 0.0:
            nerf_tex_mskd = out_nerf["tex_aux_cal_fine"]
            loss_vgg += lambda_aux * lambda_vgg * vggloss(nerf_tex_mskd, out_nerf["tar_img"])
        if loss_vgg > 0.0:
            err_dict["e_vgg"] = loss_vgg

    return err_dict

def pix_loss(src, tar, w_losses={"l1": 1.0}):
    losses = {}
    for k, v in w_losses.items():
        if v <= 0.0:
            continue
        if k == "l1":
            losses[k] = v * (src - tar).abs().mean()
        elif k == "l2":
            losses[k] = v * (src - tar).pow(2.0).mean()
        elif k == "lp":
            losses[k] = v * ((src - tar).abs() + 1e-4).pow(0.4).mean()
        elif "l1top" in k:
            ratio = float(k[5:]) / 100.0
            loss = v * (src - tar).abs().sum(1).view(src.shape[0], -1)
            loss = torch.sort(loss, dim=-1, descending=True)[0]
            loss = loss[:, :int(loss.shape[1] * ratio)]
            losses[k] = loss.mean()
        elif "l2top" in k:
            ratio = float(k[5:]) / 100.0
            loss = v * (src - tar).pow(2.0).sum(1).view(src.shape[0], -1)
            loss = torch.sort(loss, dim=-1, descending=True)[0]
            loss = loss[:, :int(loss.shape[1] * ratio)]
            losses[k] = loss.mean()
    return losses

# util NN architectures
class ResBlk(th.nn.Module):
    def __init__(self, in_ch, norm_layer):
        super(ResBlk, self).__init__()
        layers = [th.nn.ReplicationPad2d(1)]
        layers += [th.nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=0),
                   norm_layer(in_ch),
                   th.nn.ReLU(True)]

        layers += [th.nn.ReplicationPad2d(1)]
        layers += [th.nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=0),
                   norm_layer(in_ch)]

        self.layers = th.nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)

class ResBlkEncoder(th.nn.Module):
    def __init__(self, in_ch=3, out_ch=8, ngf=16, n_downsample=3, n_blocks=4, n_upsample=3, norm="instance"):
        super(ResBlkEncoder, self).__init__()
        nl = th.nn.ReLU(True)
        norm_layer = self.get_norm_layer(norm)

        layers = [th.nn.ReplicationPad2d(3), th.nn.Conv2d(
            in_ch, ngf, kernel_size=7, padding=0), norm_layer(ngf), nl]

        for i in range(n_downsample):
            mult = 2 ** i
            layers += [th.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2,
                                    padding=1), norm_layer(ngf * mult * 2), nl]

        mult = 2 ** n_downsample
        for i in range(n_blocks):
            layers += [ResBlk(ngf * mult, norm_layer)]

        for i in range(n_upsample):
            mult = 2 ** (n_downsample - i)
            layers += [th.nn.ConvTranspose2d(ngf * mult, (ngf * mult) // 2, kernel_size=3, stride=2,
                                             padding=1, output_padding=1), norm_layer((ngf * mult) // 2), nl]
        if n_upsample > 0:
            layers += [th.nn.ReplicationPad2d(3), th.nn.Conv2d((mult * ngf) // 2,
                                                               out_ch, kernel_size=7, padding=0)]

        self.layers = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def get_norm_layer(norm_type='instance'):
        if norm_type == 'batch':
            norm_layer = functools.partial(th.nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(th.nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'group':
            norm_layer = functools.partial(th.nn.GroupNorm, 16)
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

class HourGlass(torch.nn.Module):
    ''' This Module is adapted from:
        "Newell, A., Yang, K., Deng, J.: Stacked hourglass networks for human pose estimation.
        In: European conference on computer vision. pp. 483–499. Springer (2016)"
    '''

    def __init__(self, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = thf.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = thf.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class DeconvReLUGroup(torch.nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_ch, out_ch,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias)
        self.nl = torch.nn.ReLU(inplace=True)
        self.norm = torch.nn.GroupNorm(min(32, out_ch), out_ch)

    def forward(self, x):
        return self.nl(self.norm(self.conv(x)))

class HGFilterV2(torch.nn.Module):
    ''' This Module is adapted from:
        "Newell, A., Yang, K., Deng, J.: Stacked hourglass networks for human pose estimation.
        In: European conference on computer vision. pp. 483–499. Springer (2016)"
    '''
    def __init__(self, in_ch=3, out_ch=128, n_stack=2, n_downsample=4, norm="group", hd=False, **kwargs):
        super().__init__()
        self.n_stack = n_stack
        self.hd = hd

        self.nl = torch.nn.ReLU(True)

        self.unpack1 = DeconvReLUGroup(128, 32)
        self.conv_out = torch.nn.Conv2d(32, kwargs.get("out_ch_hd", 8), kernel_size=5, padding=2)

        # Base part
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if norm == 'batch':
            self.bn1 = torch.nn.BatchNorm2d(64)
        elif norm == 'group':
            self.bn1 = torch.nn.GroupNorm(32, 64)

        self.conv2 = ConvBlock(64, 128, norm)
        self.conv3 = ConvBlock(128, 128, norm)
        self.conv4 = ConvBlock(128, 256, norm)

        # Stacking part
        for hg_module in range(self.n_stack):
            self.add_module(
                'm' + str(hg_module),
                HourGlass(n_downsample, 256, norm),
            )
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, norm))
            self.add_module(
                'conv_last' + str(hg_module),
                torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            )
            if norm == 'batch':
                self.add_module('bn_end' + str(hg_module), torch.nn.BatchNorm2d(256))
            elif norm == 'group':
                self.add_module('bn_end' + str(hg_module), torch.nn.GroupNorm(32, 256))

            self.add_module(
                'l' + str(hg_module),
                torch.nn.Conv2d(256, out_ch, kernel_size=1, stride=1, padding=0),
            )

            if hg_module < self.n_stack - 1:
                self.add_module(
                    'bl' + str(hg_module),
                    torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                )
                self.add_module(
                    'al' + str(hg_module),
                    torch.nn.Conv2d(out_ch, 256, kernel_size=1, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.nl(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x_hd = self.conv_out(self.unpack1(x))

        if not self.hd:
            x = thf.avg_pool2d(x, 2, stride=2)

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.n_stack):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = thf.relu(
                self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True
            )

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.n_stack - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return [outputs[-1], x_hd]

class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()

        if norm == 'batch':
            self.bn1 = torch.nn.BatchNorm2d(in_planes)
            self.bn2 = torch.nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = torch.nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = torch.nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = torch.nn.GroupNorm(min(32, in_planes), in_planes)
            self.bn2 = torch.nn.GroupNorm(min(32, int(out_planes / 2)), int(out_planes / 2))
            self.bn3 = torch.nn.GroupNorm(min(32, int(out_planes / 4)), int(out_planes / 4))
            self.bn4 = torch.nn.GroupNorm(min(32, in_planes), in_planes)

        if in_planes != out_planes:
            self.downsample = torch.nn.Sequential(
                self.bn4,
                torch.nn.ReLU(True),
                torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

        self.conv1 = self.conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = self.conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = self.conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.nl = torch.nn.ReLU(inplace=True)

    @staticmethod
    def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
        ''' 3x3 convolution with padding. '''
        return torch.nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=strd, padding=padding, bias=bias)


    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = self.nl(out1)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = self.nl(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = self.nl(out3)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class MLPUNetFusion(th.nn.Module):
    def __init__(
        self,
        n_dims1=[28, 256, 128, 64, 32],
        n_dims2=[256, 256, 1],
        skip_dims=[256, 128, 64, 32],
        skip_layers=[1, 2, 3, 4],
        nl_layer='softplus',
        norm='weight',
        last_op=None,
        pool_types=["mean"],
        addition=False,
        dropout=False,
        pool_mode='',
        **kwargs,
    ):
        super().__init__()
        self.pool = PoolModule(
            pool_types, n_dims1[-1], pool_mode, no_sp=(n_dims1[0] == 0 and len(pool_types) * n_dims1[-1] == n_dims2[0]))

        self.layers1 = MLPUNet(
            n_dims1, skip_dims, skip_layers, nl_layer, norm, None, addition, dropout)
        self.layers2 = MLP(n_dims2, [], nl_layer, norm, last_op)

    def forward(self, x, f, a, w=None, x_add=None, nonlin=None):
        '''
        args:
            x: (B, V, N, C) Positional encoding of dz (B, V, N, 7*13)
            f: list of (B, V, N, F) Image features
            a: (B, V, N, 1) Pixel mask
            w: (B, V, N, 1) Pixel weight
        return:
            (B, M, C')
        '''
        # B, N, M, _ = a.shape
        x_view = self.layers1(x, f, nonlin)
        x_pool, valid = self.pool(x_view, a, w)  # BV
        if x_add is not None:
            x_pool = th.cat([x_pool, x_add], -1)
        out = self.layers2(x_pool, nonlin)

        return out, valid, x_view, x_pool

def get_nl_layer(nl_layer):
    nl = None
    if nl_layer == 'leakyrelu':
        nl = torch.nn.LeakyReLU(0.2, True)
    elif nl_layer == 'softplus':
        nl = torch.nn.Softplus(beta=100, threshold=20)
    elif nl_layer == 'elu':
        nl = torch.nn.ELU(inplace=True)
    elif nl_layer == 'tanh':
        nl = torch.nn.Tanh()
    elif nl_layer == 'sigmoid':
        nl = torch.nn.Sigmoid()
    elif nl_layer == "relu":
        nl = torch.nn.ReLU(inplace=True)
    elif nl_layer is not None and nl_layer not in ["none", "None", ""]:
        raise NotImplementedError(f"unsupported nl layer {nl_layer}")

    return nl

class Linear(th.nn.Module):
    def __init__(self, n_in, n_out, nonlin, wn=False):
        super().__init__()

        if wn:
            self.linear = th.nn.utils.weight_norm(th.nn.Linear(n_in, n_out))
        else:
            self.linear = th.nn.Linear(n_in, n_out)
        self.nonlin = nonlin if nonlin is not None else lambda x: x

    def forward(self, x, nonlin=None):
        x = self.linear(x)
        if nonlin is not None:
            return nonlin(x)
        else:
            return self.nonlin(x)

class MLP(nn.Module):
    def __init__(
        self,
        n_dims=[128 + 2 * 5, 256, 256, 256, 1],
        skip_layers=[2],
        nl_layer='softplus',
        norm='weight',
        last_op=None,
        **kwargs
    ):
        super(MLP, self).__init__()

        self.kwargs = kwargs
        self.last_op = get_nl_layer(last_op)
        nl = get_nl_layer(nl_layer)
        self.skip_layers = skip_layers
        self.layers = nn.ModuleList()
        for i in range(len(n_dims) - 1):
            _in, _out = n_dims[i] + n_dims[0] if i in skip_layers else n_dims[i], n_dims[i + 1]
            _nonlin = nl if i != len(n_dims) - 2 else None
            self.layers.append(Linear(_in, _out, _nonlin, norm == 'weight' and i != len(n_dims) - 2))

    def forward(self, x, nonlin=None):
        x0 = x
        for i, l in enumerate(self.layers):
            if i in self.skip_layers:
                x = th.cat([x, x0], -1)
            x = l(x, nonlin if i != len(self.layers) - 1 else None)

        if self.last_op is not None:
            return self.last_op(x)
        else:
            return x

class PoolModule(nn.Module):
    def __init__(
        self,
        pool_types,
        n_ch,
        pool_mode='',
        no_sp=False,
        n_heads=1
    ):
        super().__init__()

        self.pool_types = pool_types
        self.n_ch = n_ch
        self.pool_mode = pool_mode
        self.n_heads = n_heads
        if pool_mode == "attention_v0":
            self.proj = nn.Linear(n_ch, 1)
        elif pool_mode == "attention_v1":
            self.proj1 = nn.Linear(2 * n_ch, n_ch)
            self.proj2 = nn.Linear(n_ch, n_ch)
        # no spatial encoding
        self.no_sp = no_sp

    def forward(self, x, a, w=None):
        '''
        args:
            x: (B, V, N, C)
            a: (B, V, N, 1)
        return:
            (B, N, C)
        '''
        B, V, N, C = x.shape
        a_sum = a.sum(1)
        if w is None:
            w = a / (a_sum[:, None] + 1e-6)
        if V > 1:
            if self.pool_mode == "attention_v0":
                att = th.exp(self.proj(x))  # (B, V, N, C)
                w = w * att
                w = w / (w.sum(1, keepdim=True) + 1e-6)
            elif self.pool_mode == "attention_v1":
                D = C // self.n_heads
                q = self.proj1(pool_ops(x, ["max", "mean"], a))
                q = q.view(B, N, D, self.n_heads)
                k = self.proj2(x)
                k = k.view(B, V, N, D, self.n_heads)
                att = th.einsum('bndh,bvndh->bvnh', q, k) / (D ** 2)
                att = th.exp(att)[..., None, :].expand(-1, -1, -1, D, -1)
                w = w * att.reshape(B, V, N, -1)
                w = w / (w.sum(1, keepdim=True) + 1e-6)
                self.att = w[..., 0:1].clone()

        x = pool_ops(x, self.pool_types, w)

        if self.no_sp or self.pool_types == ["var"]:
            valid = a_sum > 1.0
        else:
            valid = a_sum > 0.0
        return x, valid

class MLPUNet(th.nn.Module):
    def __init__(
        self,
        n_dims=[28, 256, 128, 64, 32],
        skip_dims=[256, 128, 64, 32],
        skip_layers=[1, 2, 3, 4],
        nl_layer='softplus',
        norm='weight',
        last_op=None,
        addition=False,
        dropout=False,
        **kwargs,
    ):
        super().__init__()
        assert len(skip_dims) == len(skip_layers)

        self.last_op = get_nl_layer(last_op)
        self.addition = addition  # if not, use concatenation
        self.skip_layers = skip_layers
        self.skip_dict = {j: i for i, j in enumerate(skip_layers)}
        self.use_dropout = dropout
        nl = get_nl_layer(nl_layer)

        self.layers = nn.ModuleList()
        for i in range(len(n_dims) - 1):
            if self.addition or (i not in self.skip_layers):
                in_ch = n_dims[i]
            else:
                in_ch = n_dims[i] + skip_dims[self.skip_dict[i]]
            out_ch = n_dims[i + 1]

            self.layers.append(
                Linear(
                    in_ch,
                    out_ch,
                    nl if i != len(n_dims) - 2 else None,
                    norm == 'weight' and i != len(n_dims) - 2
                )
            )

        self.kwargs = kwargs

    def forward(self, x, f, nonlin=None):
        '''
        args:
            x: (B, M, C)
            feat_list: listof (B, M, Ci)
        return:
            (B, M, C')
        '''
        assert isinstance(f, List)

        if self.training and self.use_dropout:
            drop_idx = random.randint(0, 2 * len(self.layers))
        else:
            drop_idx = len(self.layers)
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                x1 = f[self.skip_dict[i]]
                if i > drop_idx:
                    x1 *= 0.0

                if x is not None:
                    x = x + x1 if self.addition else th.cat([x, x1], -1)
                else:
                    x = x1
            x = self.layers[i](x, nonlin if i != len(self.layers) - 1 else None)

        if self.last_op is not None:
            return self.last_op(x)
        else:
            return x

def pool_ops(x, pool_types, w=None):
    '''
    args:
        x: (B, V, N, C)
    '''
    ret = []
    if "max" in pool_types:
        val = x.max(1)[0]
        ret.append(val)
    if any(f in pool_types for f in ["mean", "var"]):
        if w is not None:
            wx = w * x
            mean = wx.sum(1)
        else:
            mean = x.mean(1)
        if "mean" in pool_types:
            ret.append(mean)
        if "var" in pool_types:
            if w is not None:
                var = w * (x - mean[:, None]).pow(2.0)
                var = var.sum(1)
            else:
                var = (x - mean[:, None]).pow(2.0)
                var = var.mean(1)
            ret.append(var)

    return th.cat(ret, -1)

class Vgg19(th.nn.Module):
    ''' This Module is adapted from:
        "Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale
        image recognition. arXiv preprint arXiv:1409.1556 (2014)"
    '''
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = th.nn.Sequential()
        self.slice2 = th.nn.Sequential()
        self.slice3 = th.nn.Sequential()
        self.slice4 = th.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)

        out = [h_relu1, h_relu2, h_relu3, h_relu4]
        return out

class VGGLoss(th.nn.Module):
    ''' This Module is adapted from:
        "Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale
        image recognition. arXiv preprint arXiv:1409.1556 (2014)"
    '''
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg_net = Vgg19().cuda()
        self.l1_loss = th.nn.L1Loss()
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x, y = self.normalize(x), self.normalize(y)
        x_vgg, y_vgg = self.vgg_net(x), self.vgg_net(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.l1_loss(x_vgg[i], y_vgg[i].detach())
        return loss