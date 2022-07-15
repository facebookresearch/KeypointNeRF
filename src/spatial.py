# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np

class SpatialEncoder(torch.nn.Module):
    def __init__(self, sp_level, sp_type, scale, n_kpt, **kwargs):
        super().__init__()

        self.sp_type = sp_type
        self.sp_level = sp_level
        self.n_kpt = n_kpt
        self.scale = scale

        center = torch.Tensor(kwargs.get("center", [0.0, 0.0, 0.0])).float()
        self.register_buffer("center", center)

        self.kwargs = kwargs

    @staticmethod
    def position_embedding(x, nlevels, scale=1.0):
        '''
        args:
            x: (B, N, C)
        return:
            (B, N, C * n_levels * 2)
        '''
        if nlevels <= 0:
            return x
        vec = SpatialEncoder.pe_vector(nlevels, x.device, scale)

        B, N, _ = x.shape
        y = x[:, :, None, :] * vec[None, None, :, None]
        z = torch.cat((torch.sin(y), torch.cos(y)), axis=-1).view(B, N, -1)

        return torch.cat([x, z], -1)

    @staticmethod
    def pe_vector(nlevels, device, scale=1.0):
        v, val = [], 1
        for _ in range(nlevels):
            v.append(scale * np.pi * val)
            val *= 2
        return torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)

    def get_dim(self):
        if self.sp_type in ["z", "rel_z", "rel_z_decay"]:
            if "rel" in self.sp_type:
                return (1 + 2 * self.sp_level) * self.n_kpt
            else:
                return 1 + 2 * self.sp_level
        elif "xyz" in self.sp_type:
            if "rel" in self.sp_type:
                return (1 + 2 * self.sp_level) * 3 * self.n_kpt
            else:
                return (1 + 2 * self.sp_level) * 3

        return 0

    def forward(self, **sp_data):
        KRT = sp_data['KRT']
        v = sp_data['v']  # after view expansion
        vo = sp_data['pts']  # before view expansion

        V = sp_data['n_view']
        B = KRT.shape[0] // V
        N = vo.shape[-2]

        z = sp_data['z']
        xy = sp_data['xy']

        Rt = sp_data['extrin']
        cxyz = v @ Rt[:, :3, :3].transpose(1, 2) + Rt[:, :3, 3][:, None]

        if "rel" in self.sp_type:
            kpt3do = sp_data['kpt3d']  # before view expansion
            assert kpt3do.shape[1] == self.n_kpt
            kpt3d = kpt3do[:, None].expand(-1, V, -1, -1)
            kpt3d = kpt3d.reshape(-1, *kpt3d.shape[2:])  # after view expansion

            Rt = sp_data['extrin']
            kptxyz = kpt3d @ Rt[:, :3, :3].transpose(1, 2) + Rt[:, :3, 3][:, None]

        out = None
        if self.sp_type == "z":
            out = self.position_embedding(z, self.sp_level)
        elif self.sp_type == "ixyz":  # image space
            xyz = torch.cat([xy, z], -1)
            out = self.position_embedding(xyz, self.sp_level)
        elif self.sp_type == "cxyz":  # camera space
            out = self.position_embedding(cxyz, self.sp_level)
        elif self.sp_type == "mxyz":  # model space
            T = sp_data['T']
            mxyz = self.scale * (vo @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3][:, None])
            out = self.position_embedding(mxyz, self.sp_level)
            if self.kwargs.get("view_expand", True):
                out = out[:, None].expand(-1, V, -1, -1).reshape(B * V, N, -1)
        elif self.sp_type == "wxyz":  # world space
            if self.kwargs.get("view_expand", True):
                out = self.scale * (v - self.center[None, None])
            else:
                out = self.scale * (vo - self.center[None, None])
            out = self.position_embedding(out, self.sp_level)
        elif self.sp_type == "rel_z":
            dz = self.scale * (cxyz[:, :, None, 2:3] - kptxyz[:, None, :, 2:3])
            out = self.position_embedding(dz.view(*dz.shape[:2], -1), self.sp_level)
        elif self.sp_type == "rel_z_decay":
            dz = self.scale * (cxyz[:, :, None, 2:3] - kptxyz[:, None, :, 2:3])
            sigma = self.kwargs.get('sigma', 150.0)
            dxyz = cxyz[:, :, None] - kptxyz[:, None, :]
            w = torch.exp(-(dxyz**2).sum(-1, keepdim=True) / (2.0 * (sigma ** 2)))
            w = w.view(*w.shape[:2], -1) # (B, N, K)
            out = self.position_embedding(dz.view(*dz.shape[:2], -1), self.sp_level)
            out = out.view(*out.shape[:2], -1, w.shape[-1]) * w[:, :, None]  # BV,N,7,13 * BV,N,1,13
            out = out.view(*out.shape[:2], -1)
        elif self.sp_type == "rel_cxyz":
            dxyz = self.scale * (cxyz[:, :, None] - kptxyz[:, None])
            out = self.position_embedding(dxyz.view(*dxyz.shape[:2], -1), self.sp_level)
        elif self.sp_type == "rel_wxyz":
            dxyz = v[:, :, None] - kpt3d[:, None]
            out = self.position_embedding(dxyz.reshape(*dxyz.shape[:2], -1), self.sp_level)
        elif self.sp_type == "rel_mxyz":
            T = sp_data['T']
            mxyz = self.scale * (vo @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3][:, None])
            kmxyz = self.scale * (vpt3do @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3][:, None])
            dxyz = mxyz[:, :, None] - kmxyz[:, None]
            out = self.position_embedding(
                dxyz.view(*dxyz.shape[:2], -1), self.sp_level)
            if self.kwargs.get("view_expand", True):
                out = out[:, None].expand(-1, V, -1, -1).reshape(B * V, N, -1)

        return out
