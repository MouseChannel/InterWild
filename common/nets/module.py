# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import math

import kornia
import torch
import torch.nn as nn
from main.config import cfg
from common.nets.layer import make_conv_layers, make_deconv_layers, make_linear_layers
from torch.nn import functional as F
from common.utils.mano import mano
from common.utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d


class MouseChannelSelect(torch.autograd.Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def symbolic(g, input, one_hot):
        """Symbolic function for creating onnx op."""
        return g.op(
            'com.microsoft::MouseChannelSelect',
            input,
            one_hot)
        

    @staticmethod
    def forward(g, input, one_hot):
        origin = input[one_hot > 0.5, :]
        

        return origin


class MouseChannelSmallAngle(torch.autograd.Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def symbolic(g, sin_half_angles_over_angles, small_angles, angles):
        """Symbolic function for creating onnx op."""
        return g.op(
            'com.microsoft::MouseChannelSmallAngle',
            sin_half_angles_over_angles,
            small_angles,
            angles)
       

    @staticmethod
    def forward(g, sin_half_angles_over_angles, small_angles, angles):
        # small_angles = torch.where(small_angles > 0.5, True, False)
        small_angles = torch.where(small_angles > 0.5, torch.tensor(True), torch.tensor(False))

        sin_half_angles_over_angles[~small_angles] = (
                torch.sin((angles / 2)[~small_angles]) / angles[~small_angles]
        )

        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
                0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        return sin_half_angles_over_angles


def linalg_inv_settype(g, self):
    return (g.op("com.microsoft::MouseChannelInverse", self)
    .setType(
        self.type().with_dtype(torch.float).with_sizes([None, 3, 3])
    ))
torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_settype, 17)




class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.depth_dim = cfg.output_hand_hm_shape[0]
        self.conv = make_conv_layers([2048, self.joint_num * self.depth_dim], kernel=1, stride=1, padding=0,
                                     bnrelu_final=False)

    def forward(self, hand_feat):
        hand_hm = self.conv(hand_feat)
        _, _, height, width = hand_hm.shape
        hand_hm = hand_hm.view(-1, self.joint_num, self.depth_dim, height, width)
        # return hand_hm
        hand_coord = soft_argmax_3d(hand_hm)
        return hand_coord


class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.conv = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.root_pose_out = make_linear_layers([21 * (512 + 3), 6], relu_final=False)
        self.pose_out = make_linear_layers([21 * (512 + 3), (mano.orig_joint_num - 1) * 6],
                                           relu_final=False)  # without root joint
        self.shape_out = make_linear_layers([2048, mano.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([2048, 3], relu_final=False)

    def get_root_trans(self, cam_param):
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_hand_shape[0] * cfg.input_hand_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        root_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return root_trans

    def rotation_6d_to_matrix(self, d6: torch.Tensor) -> torch.Tensor:
         

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = self._sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                    ],
                dim=-1,
            )
        )

        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = torch.stack(
            [
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)
        # tt = F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
        one_hot = F.one_hot(q_abs.argmax(dim=-1), num_classes=4).float()
      

        data = MouseChannelSelect.apply(quat_candidates, one_hot)
      
        return data
         

    def _sqrt_positive_part(self, x: torch.Tensor) -> torch.Tensor:
        
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
       
        return ret

    def matrix_to_axis_angle(self, matrix: torch.Tensor) -> torch.Tensor:
        
        return self.quaternion_to_axis_angle(self.matrix_to_quaternion(matrix))

    def quaternion_to_axis_angle(self, quaternions: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)

        half_angles = torch.atan2(norms, quaternions[..., :1])
        angles = 2 * half_angles
        eps = 1e-6
        # small_angles = angles.abs() < eps
        small_angles = torch.where(angles.abs() < eps, torch.tensor(1), torch.tensor(0))
        # aa = ~small_angles
        sin_half_angles_over_angles = torch.empty_like(angles)

        temp = MouseChannelSmallAngle.apply(sin_half_angles_over_angles, small_angles, angles)
        return quaternions[..., 1:] / temp
        small_angles = torch.where(small_angles > 0.5, True, False)
        sin_half_angles_over_angles[~small_angles] = (
                torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )

        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
                0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )

        return quaternions[..., 1:] / sin_half_angles_over_angles

    def forward(self, hand_feat, joint_img):
        batch_size = hand_feat.shape[0]

        # shape and root translation
        shape_param = self.shape_out(hand_feat.mean((2, 3)))
        cam_param = self.cam_out(hand_feat.mean((2, 3)))
        root_trans = self.get_root_trans(cam_param)

        # pose
        hand_feat = self.conv(hand_feat)
        hand_feat = sample_joint_features(hand_feat, joint_img[:, :, :2])  # batch_size, joint_num, feat_dim
        hand_feat = torch.cat((hand_feat, joint_img), 2).view(batch_size, -1)
        root_pose = self.root_pose_out(hand_feat)

        # root_pose = self.matrix_to_quaternion(self.rotation_6d_to_matrix(root_pose))

        root_pose = self.quaternion_to_axis_angle(self.matrix_to_quaternion(self.rotation_6d_to_matrix(root_pose)))

        hand_pose = self.pose_out(hand_feat).view(batch_size, -1, 6)
        hand_pose = self.quaternion_to_axis_angle(self.matrix_to_quaternion(self.rotation_6d_to_matrix(hand_pose))).reshape(batch_size, -1)
        
        return root_pose, hand_pose, shape_param, root_trans


class TransNet(nn.Module):
    def __init__(self, backbone):
        super(TransNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.backbone = backbone
        self.conv = make_conv_layers([self.joint_num * 2 * cfg.input_hm_shape[0], 64], kernel=3, stride=1, padding=1)
        self.fc = make_linear_layers([2 * (512 + 2), 3], relu_final=False)

    def get_bbox(self, joint_img):
        x_img, y_img = joint_img[:, :, 0], joint_img[:, :, 1]
        xmin = torch.min(x_img, 1)[0];
        ymin = torch.min(y_img, 1)[0];
        xmax = torch.max(x_img, 1)[0];
        ymax = torch.max(y_img, 1)[0];

        x_center = (xmin + xmax) / 2.;
        width = xmax - xmin;
        xmin = x_center - 0.5 * width * 1.2
        xmax = x_center + 0.5 * width * 1.2

        y_center = (ymin + ymax) / 2.;
        height = ymax - ymin;
        ymin = y_center - 0.5 * height * 1.2
        ymax = y_center + 0.5 * height * 1.2

        bbox = torch.stack((xmin, ymin, xmax, ymax), 1).float().cuda()
        return bbox

    def render_gaussian_heatmap(self, joint_coord):
        x = torch.arange(cfg.input_hm_shape[2])
        y = torch.arange(cfg.input_hm_shape[1])
        z = torch.arange(cfg.input_hm_shape[0])
        zz, yy, xx = torch.meshgrid(z, y, x)
        xx = xx[None, None, :, :, :].cuda().float();
        yy = yy[None, None, :, :, :].cuda().float();
        zz = zz[None, None, :, :, :].cuda().float();

        x = joint_coord[:, :, 0, None, None, None];
        y = joint_coord[:, :, 1, None, None, None];
        z = joint_coord[:, :, 2, None, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
        heatmap = heatmap * 255
        return heatmap

    def set_aspect_ratio(self, bbox, aspect_ratio=1):
        # xyxy -> xywh
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

        w = bbox[:, 2]
        h = bbox[:, 3]
        c_x = bbox[:, 0] + w / 2.
        c_y = bbox[:, 1] + h / 2.

        # aspect ratio preserving bbox
        mask1 = w > (aspect_ratio * h)
        mask2 = w < (aspect_ratio * h)
        h[mask1] = w[mask1] / aspect_ratio
        w[mask2] = h[mask2] * aspect_ratio

        bbox[:, 2] = w
        bbox[:, 3] = h
        bbox[:, 0] = c_x - bbox[:, 2] / 2.
        bbox[:, 1] = c_y - bbox[:, 3] / 2.

        # xywh -> xyxy
        bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
        return bbox

    def crop_and_resize(self, coord, bbox):
        batch_size = bbox.shape[0]
        transl = torch.FloatTensor([cfg.input_hm_shape[2] / 2, cfg.input_hm_shape[1] / 2]).cuda()[None, :] - (
                bbox[:, :2] + bbox[:, 2:]) / 2
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        scale = torch.stack([cfg.input_hm_shape[2] / (bbox[:, 2] - bbox[:, 0] + 1e-8),
                             cfg.input_hm_shape[1] / (bbox[:, 3] - bbox[:, 1] + 1e-8)], 1)
        angle = torch.zeros((batch_size)).float().cuda()
        orig2hm_trans = kornia.geometry.transform.get_affine_matrix2d(transl, center, scale, angle)[:, :2, :]

        xy1 = torch.stack((coord[:, :, 0], coord[:, :, 1], torch.ones_like(coord[:, :, 0])), 2)
        xy = torch.bmm(orig2hm_trans, xy1.permute(0, 2, 1)).permute(0, 2, 1)
        coord = torch.cat((xy, coord[:, :, 2:]), 2)
        return coord

    def forward(self, rjoint_img, ljoint_img, rhand_hand2orig_trans, lhand_hand2orig_trans):
        rjoint_img, ljoint_img = rjoint_img.clone(), ljoint_img.clone()

        # cfg.output_hand_hm_shape -> cfg.input_img_shape
        for coord, trans in ((ljoint_img, lhand_hand2orig_trans), (rjoint_img, rhand_hand2orig_trans)):
            x, y = coord[:, :, 0], coord[:, :, 1]
            x = x / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
            y = y / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]
            xy1 = torch.stack((x, y, torch.ones_like(x)), 2)
            xy = torch.bmm(trans, xy1.permute(0, 2, 1)).permute(0, 2, 1)
            coord[:, :, 0] = xy[:, :, 0]
            coord[:, :, 1] = xy[:, :, 1]

        # compute tight hand bboxes from joint_img
        rhand_bbox = self.get_bbox(rjoint_img)  # xmin, ymin, xmax, ymax in cfg.input_img_shape
        lhand_bbox = self.get_bbox(ljoint_img)  # xmin, ymin, xmax, ymax in cfg.input_img_shape

        # bbox union
        xmin = torch.minimum(lhand_bbox[:, 0], rhand_bbox[:, 0])
        ymin = torch.minimum(lhand_bbox[:, 1], rhand_bbox[:, 1])
        xmax = torch.maximum(lhand_bbox[:, 2], rhand_bbox[:, 2])
        ymax = torch.maximum(lhand_bbox[:, 3], rhand_bbox[:, 3])
        hand_bbox_union = torch.stack((xmin, ymin, xmax, ymax), 1)
        hand_bbox_union = self.set_aspect_ratio(hand_bbox_union)

        # cfg.input_img_shape -> cfg.input_hm_shape
        rjoint_img = self.crop_and_resize(rjoint_img, hand_bbox_union)
        ljoint_img = self.crop_and_resize(ljoint_img, hand_bbox_union)
        rjoint_img[:, :, 2] = rjoint_img[:, :, 2] / cfg.output_hand_hm_shape[0] * cfg.input_hm_shape[0]
        ljoint_img[:, :, 2] = ljoint_img[:, :, 2] / cfg.output_hand_hm_shape[0] * cfg.input_hm_shape[0]

        # hand heatmap
        rhand_hm = self.render_gaussian_heatmap(rjoint_img)
        rhand_hm = rhand_hm.view(-1, self.joint_num * cfg.input_hm_shape[0], cfg.input_hm_shape[1],
                                 cfg.input_hm_shape[2])

        # rhand_hm = rhand_hm.view(int(rhand_hm.numel() / (self.joint_num*cfg.input_hm_shape[0]) / cfg.input_hm_shape[1] /cfg.input_hm_shape[2]),self.joint_num*cfg.input_hm_shape[0],cfg.input_hm_shape[1],cfg.input_hm_shape[2])

        rhand_hm = rhand_hm.view((rhand_hm.numel() / (self.joint_num * cfg.input_hm_shape[0]) / cfg.input_hm_shape[1] /
                                  cfg.input_hm_shape[2]), self.joint_num * cfg.input_hm_shape[0], cfg.input_hm_shape[1],
                                 cfg.input_hm_shape[2])

        lhand_hm = self.render_gaussian_heatmap(ljoint_img)
        # lhand_hm = lhand_hm.view(-1,self.joint_num*cfg.input_hm_shape[0],cfg.input_hm_shape[1],cfg.input_hm_shape[2])
        lhand_hm = lhand_hm.view((lhand_hm.numel() / (self.joint_num * cfg.input_hm_shape[0]) / cfg.input_hm_shape[1] /
                                  cfg.input_hm_shape[2]), self.joint_num * cfg.input_hm_shape[0], cfg.input_hm_shape[1],
                                 cfg.input_hm_shape[2])
        hand_hm = torch.cat((rhand_hm, lhand_hm), 1)

        # relative translation
        hand_feat = self.backbone(self.conv(hand_hm), stage='late')
        wrist_img = torch.stack((rjoint_img[:, mano.sh_root_joint_idx, :], ljoint_img[:, mano.sh_root_joint_idx, :]), 1)
        wrist_img = torch.stack((wrist_img[:, :, 0] / 8, wrist_img[:, :, 1] / 8), 2)
        wrist_feat = sample_joint_features(hand_feat, wrist_img)
        wrist_feat = torch.cat((wrist_feat, wrist_img), 2).view(-1, 2 * (512 + 2))
        rel_trans = self.fc(wrist_feat)
        return rel_trans


class BoxNet(nn.Module):
    def __init__(self):
        super(BoxNet, self).__init__()
        self.deconv = make_deconv_layers([2048, 256, 256, 256])
        self.bbox_center = make_conv_layers([256, 2], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)

    def get_conf(self, bbox_center_hm, rhand_center, lhand_center):
        batch_size = bbox_center_hm.shape[0]

        bbox_center_hm = bbox_center_hm.view(batch_size, 2, cfg.output_body_hm_shape[1] * cfg.output_body_hm_shape[2])
        bbox_center_hm = F.softmax(bbox_center_hm, 2)
        bbox_center_hm = bbox_center_hm.view(batch_size, 2, cfg.output_body_hm_shape[1], cfg.output_body_hm_shape[2])

        rhand_conf = sample_joint_features(bbox_center_hm[:, 0, None, :, :], rhand_center[:, None, :])[:, 0, 0]
        lhand_conf = sample_joint_features(bbox_center_hm[:, 1, None, :, :], lhand_center[:, None, :])[:, 0, 0]
        return rhand_conf, lhand_conf

    def forward(self, img_feat):
        img_feat = self.deconv(img_feat)

        # bbox center
        bbox_center_hm = self.bbox_center(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        rhand_center, lhand_center = bbox_center[:, 0, :], bbox_center[:, 1, :]

        # bbox size
        rhand_feat = sample_joint_features(img_feat, rhand_center.detach()[:, None, :])[:, 0, :]
        rhand_size = self.rhand_size(rhand_feat)
        lhand_feat = sample_joint_features(img_feat, lhand_center.detach()[:, None, :])[:, 0, :]
        lhand_size = self.lhand_size(lhand_feat)

        # bbox conf
        rhand_conf, lhand_conf = self.get_conf(bbox_center_hm, rhand_center, lhand_center)

        # rhand_center = torch.rand(1,2).cuda()
        # rhand_size = torch.rand(1,2).cuda()
        # lhand_center = torch.rand(1,2).cuda()
        # lhand_size = torch.rand(1,2).cuda()

        # rhand_conf = torch.rand(1).cuda()
        # lhand_conf = torch.rand(1).cuda()

        return rhand_center, rhand_size, lhand_center, lhand_size, rhand_conf, lhand_conf


class HandRoI(nn.Module):
    def __init__(self, backbone):
        super(HandRoI, self).__init__()
        self.backbone = backbone

    def affine_grid(self, theta, size, align_corners=False):
        N, C, H, W = size
        grid = self.create_grid(N, C, H, W)
        grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
        grid = grid.view(N, H, W, 2)
        return grid

    def create_grid(self, N, C, H, W):
        grid = torch.empty((N, H, W, 3), dtype=torch.float32).cuda()
        grid.select(-1, 0).copy_(self.linspace_from_neg_one(W))
        grid.select(-1, 1).copy_(self.linspace_from_neg_one(H).unsqueeze_(-1))
        grid.select(-1, 2).fill_(1)
        return grid

    def linspace_from_neg_one(self, num_steps, dtype=torch.float32):
        r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
        # r = r * (num_steps - 1) / num_steps
        return r

    def normalize_homography(self, dst_pix_trans_src_pix, dsize_src: tuple[int, int], dsize_dst: tuple[int, int]):
        src_h, src_w = dsize_src
        dst_h, dst_w = dsize_dst

        # compute the transformation pixel/norm for src/dst
        src_norm_trans_src_pix = self.normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

        src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
        dst_norm_trans_dst_pix = self.normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

        # compute chain transformations
        dst_norm_trans_src_norm = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
        return dst_norm_trans_src_norm

    def normal_transform_pixel(self, height, width, eps=1e-9):
        tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]).float().cuda()  # 3x3

       
        width_denom = eps if width == 1 else width - 1.0
        height_denom = eps if height == 1 else height - 1.0

        tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
        tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
        return tr_mat.unsqueeze(0)

    def warp_affine(self, src, M, dsize):
        B, C, H, W = src.size()
        M3 = F.pad(M, [0, 0, 0, 1], "constant", value=0.0)
        # H = torch.rand(1,3,3).cuda()
        M3[..., -1, -1] += 1.0

        dst_norm_trans_src_norm = self.normalize_homography(M3, (H, W), dsize)
        src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
        grid = self.affine_grid(src_norm_trans_dst_norm[:, :2, :], [B, C, dsize[0], dsize[1]], align_corners=True)
        return F.grid_sample(src, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

    def crop_and_resize(self, img, bbox):
        batch_size = bbox.shape[0]
        bbox = bbox.clone()  # xmin, ymin, xmax, ymax in cfg.input_body_shape space
        bbox[:, 0] = bbox[:, 0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        bbox[:, 1] = bbox[:, 1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        bbox[:, 2] = bbox[:, 2] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        bbox[:, 3] = bbox[:, 3] / cfg.input_body_shape[0] * cfg.input_img_shape[0]

        

        transl = torch.FloatTensor([cfg.input_hand_shape[1] / 2, cfg.input_hand_shape[0] / 2]).cuda()[None, :] - (
                bbox[:, :2] + bbox[:, 2:]) / 2
      
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        scale = torch.stack([cfg.input_hand_shape[1] / (bbox[:, 2] - bbox[:, 0] + 1e-8),
                             cfg.input_hand_shape[0] / (bbox[:, 3] - bbox[:, 1] + 1e-8)], 1)
        
        angle = torch.zeros((1)).float().cuda()
       
        orig2hand_trans = kornia.geometry.transform.get_affine_matrix2d(transl, center, scale, angle)

      
        hand2orig_trans = torch.inverse(orig2hand_trans)[:, :2, :]

        orig2hand_trans = orig2hand_trans[:, :2, :]

       
        img = self.warp_affine(img, orig2hand_trans, (256, 256))

       
        return img, orig2hand_trans, hand2orig_trans

    def forward(self, img, rhand_bbox, lhand_bbox):
        # cfg.input_img_shape -> cfg.input_hand_shape
        with torch.no_grad():
            rhand_img, rhand_orig2hand_trans, rhand_hand2orig_trans = self.crop_and_resize(img, rhand_bbox)
            lhand_img, lhand_orig2hand_trans, lhand_hand2orig_trans = self.crop_and_resize(img, lhand_bbox)
            lhand_img = torch.flip(lhand_img, [3])  # flip to the right hand

        hand_img = torch.cat((rhand_img, lhand_img))
        hand_feat = self.backbone(hand_img)
        orig2hand_trans = torch.cat((rhand_orig2hand_trans, lhand_orig2hand_trans))
        hand2orig_trans = torch.cat((rhand_hand2orig_trans, lhand_hand2orig_trans))
        return hand_feat, orig2hand_trans, hand2orig_trans
