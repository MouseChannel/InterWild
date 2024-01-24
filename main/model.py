# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.nets.loss import CoordLoss, PoseLoss
from common.nets.module import BoxNet, HandRoI, PositionNet, RotationNet, TransNet
from common.nets.resnet import ResNetBackbone
from common.utils.mano import mano
from common.utils.transforms import restore_bbox

from main.config import cfg


class Model(nn.Module):
    def __init__(self, body_backbone, body_box_net, hand_roi_net, hand_position_net, hand_rotation_net, hand_trans_net):
        super(Model, self).__init__()
        self.body_backbone = body_backbone
        self.body_box_net = body_box_net

        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net
        self.hand_trans_net = hand_trans_net



        self.mano_layer_right = mano.right_mano_layer.cuda()
        self.mano_layer_left = mano.left_mano_layer.cuda()
 
        self.coord_loss = CoordLoss()
        self.pose_loss = PoseLoss()

        self.trainable_modules = [self.body_backbone, self.body_box_net, self.hand_roi_net, self.hand_position_net,
                                  self.hand_rotation_net, self.hand_trans_net]

    def get_coord(self, root_pose, hand_pose, shape, root_trans, hand_type):
        batch_size = root_pose.shape[0]
        zero_trans = torch.zeros((batch_size, 3)).float().cuda()
        if hand_type == 'right':
            output = self.mano_layer_right(betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=zero_trans)
        else:
            output = self.mano_layer_left(betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=zero_trans)
      
        mesh_cam = output
 
        joint_cam = torch.bmm(torch.from_numpy(mano.sh_joint_regressor).cuda()[None, :, :].repeat(batch_size, 1, 1),
                              mesh_cam)
        root_cam = joint_cam[:, mano.sh_root_joint_idx, :]
        mesh_cam = mesh_cam - root_cam[:, None, :] + root_trans[:, None, :]
        joint_cam = joint_cam - root_cam[:, None, :] + root_trans[:, None, :]

        # project 3D coordinates to 2D space
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        y = y / cfg.input_hand_shape[0] * cfg.output_hand_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, mano.sh_root_joint_idx, :]
        joint_cam = joint_cam - root_cam[:, None, :]
        mesh_cam = mesh_cam - root_cam[:, None, :]
        return joint_proj, joint_cam, mesh_cam, root_cam

  
     
    

  

    def forward(self, input_img):
 
        body_img = F.interpolate(input_img, cfg.input_body_shape, mode='bilinear')

        body_feat = self.body_backbone(body_img)

        rhand_bbox_center, rhand_bbox_size, lhand_bbox_center, lhand_bbox_size, rhand_bbox_conf, lhand_bbox_conf = self.body_box_net(
            body_feat)
  
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
                                  2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
                                  2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

       
        hand_feat, orig2hand_trans, hand2orig_trans = self.hand_roi_net(input_img, rhand_bbox,
                                                                        lhand_bbox)  # (2N, ...). right hand + flipped left hand




        # hand network
        joint_img = self.hand_position_net(hand_feat)

        mano_root_pose, mano_hand_pose, mano_shape, root_trans = self.hand_rotation_net(hand_feat, joint_img.detach())
        # mano_test = self.hand_rotation_net(hand_feat, joint_img.detach())

 
        rhand_num, lhand_num = rhand_bbox.shape[0], lhand_bbox.shape[0]

       
        rjoint_img = joint_img[:rhand_num, :, :]

        ljoint_img = joint_img[rhand_num:, :, :]
        ljoint_img_x = ljoint_img[:, :, 0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        ljoint_img_x = cfg.input_hand_shape[1] - 1 - ljoint_img_x
        ljoint_img_x = ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        ljoint_img = torch.cat((ljoint_img_x[:, :, None], ljoint_img[:, :, 1:]), 2)
        # restore flipped left root rotations
        rroot_pose = mano_root_pose[:rhand_num, :]
        lroot_pose = mano_root_pose[rhand_num:, :]
        lroot_pose = torch.cat((lroot_pose[:, 0:1], -lroot_pose[:, 1:3]), 1)
        # restore flipped left hand joint rotations
        rhand_pose = mano_hand_pose[:rhand_num, :]
        lhand_pose = mano_hand_pose[rhand_num:, :].reshape(-1, mano.orig_joint_num-1, 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(lhand_num, -1)
        # shape
        rshape = mano_shape[:rhand_num, :]
        lshape = mano_shape[rhand_num:, :]
        # restore flipped left root translation
        rroot_trans = root_trans[:rhand_num, :]
        lroot_trans = root_trans[rhand_num:, :]
        lroot_trans = torch.cat((-lroot_trans[:, 0:1], lroot_trans[:, 1:]), 1)
        # affine transformation matrix
        rhand_orig2hand_trans = orig2hand_trans[:rhand_num]
        lhand_orig2hand_trans = orig2hand_trans[rhand_num:]
        rhand_hand2orig_trans = hand2orig_trans[:rhand_num]
        lhand_hand2orig_trans = hand2orig_trans[rhand_num:]


 
        rjoint_proj, rjoint_cam, rmesh_cam, rroot_cam = self.get_coord(rroot_pose, rhand_pose, rshape, rroot_trans,
                                                                       'right')
        ljoint_proj, ljoint_cam, lmesh_cam, lroot_cam = self.get_coord(lroot_pose, lhand_pose, lshape, lroot_trans,
                                                                       'left')

       
        mesh_cam = torch.cat((rmesh_cam, lmesh_cam), 1)
       
        joint_img = torch.cat((rjoint_img, ljoint_img), 1)

        joint_proj = torch.cat((rjoint_proj, ljoint_proj), 1)

       
        part_name = 'right'
        trans = rhand_hand2orig_trans
        x = joint_proj[:, mano.th_joint_type[part_name], 0] / cfg.output_hand_hm_shape[2] * \
            cfg.input_hand_shape[1]
        y = joint_proj[:, mano.th_joint_type[part_name], 1] / cfg.output_hand_hm_shape[1] * \
            cfg.input_hand_shape[0]

        xy1 = torch.stack((x, y, torch.ones_like(x)), 2)
        xy = torch.bmm(trans, xy1.permute(0, 2, 1)).permute(0, 2, 1)
        joint_proj[:, mano.th_joint_type[part_name], 0] = xy[:, :, 0]
        joint_proj[:, mano.th_joint_type[part_name], 1] = xy[:, :, 1]

        
        x = joint_img[:, mano.th_joint_type[part_name], 0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[
            1]
        y = joint_img[:, mano.th_joint_type[part_name], 1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[
            0]

        xy1 = torch.stack((x, y, torch.ones_like(x).cuda()), 2)
        xy = torch.bmm(trans, xy1.permute(0, 2, 1)).permute(0, 2, 1)

    
        joint_img[:, :21, 0] = xy[:, :, 0]
        joint_img[:, :21, 1] = xy[:, :, 1]

        ###
        part_name = 'left'
        trans = lhand_hand2orig_trans
        x = joint_proj[:, mano.th_joint_type[part_name], 0] / cfg.output_hand_hm_shape[2] * \
            cfg.input_hand_shape[1]
        y = joint_proj[:, mano.th_joint_type[part_name], 1] / cfg.output_hand_hm_shape[1] * \
            cfg.input_hand_shape[0]

        xy1 = torch.stack((x, y, torch.ones_like(x)), 2)
        xy = torch.bmm(trans, xy1.permute(0, 2, 1)).permute(0, 2, 1)
        joint_proj[:, mano.th_joint_type[part_name], 0] = xy[:, :, 0]
        joint_proj[:, mano.th_joint_type[part_name], 1] = xy[:, :, 1]

   
        x = joint_img[:, mano.th_joint_type[part_name], 0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[
            1]
        y = joint_img[:, mano.th_joint_type[part_name], 1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[
            0]

        xy1 = torch.stack((x, y, torch.ones_like(x).cuda()), 2)
        xy = torch.bmm(trans, xy1.permute(0, 2, 1)).permute(0, 2, 1)

        
        joint_img[:, 21:, 0] = xy[:, :, 0]
        joint_img[:, 21:, 1] = xy[:, :, 1]

      

        # test output
        out = {}
        

        out['rjoint_img'] = joint_img[:, :21, :]

        out['ljoint_img'] = joint_img[:, 21:, :]
        

        out['rmano_mesh_cam'] = rmesh_cam
        out['lmano_mesh_cam'] = lmesh_cam
       
        return out


def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass


def linspace_from_neg_one(num_steps, dtype=torch.float32):
    r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
    # r = r * (num_steps - 1) / num_steps
    return r


def normalize_homography(dst_pix_trans_src_pix, dsize_src: tuple[int, int], dsize_dst: tuple[int, int]):
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def normal_transform_pixel(height, width, eps=1e-9):
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]).float().cuda()  # 3x3

    # prevent divide by zero bugs
    # if width == 1 :
    #     width_denom = eps
    # else:
    #     width_denom =width - 1.0
    # if height == 1 :
    #     height_denom = eps
    # else:
    #     height_denom =height - 1.0
    width_denom = eps if width == 1 else width - 1.0
    height_denom = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    return tr_mat.unsqueeze(0)


def affine_grid(theta, size, align_corners=False):
    N, C, H, W = size
    grid = create_grid(N, C, H, W)
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2)
    return grid


def create_grid(N, C, H, W):
    grid = torch.empty((N, H, W, 3), dtype=torch.float32).cuda()
    grid.select(-1, 0).copy_(linspace_from_neg_one(W))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1)
    return grid


def warp_affine(src, M, dsize):
    B, C, H, W = src.size()
    M3 = F.pad(M, [0, 0, 0, 1], "constant", value=0.0)
    # H = torch.rand(1,3,3).cuda()
    M3[..., -1, -1] += 1.0

    dst_norm_trans_src_norm = normalize_homography(M3, (H, W), dsize)
    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    grid = affine_grid(src_norm_trans_dst_norm[:, :2, :], [B, C, dsize[0], dsize[1]], align_corners=True)
    return F.grid_sample(src, grid, align_corners=True, mode='bilinear', padding_mode='zeros')


def get_model(mode):
    body_backbone = ResNetBackbone(cfg.body_resnet_type)
    body_box_net = BoxNet()

    hand_backbone = ResNetBackbone(cfg.hand_resnet_type)
    hand_roi_net = HandRoI(hand_backbone)
    hand_position_net = PositionNet()
    hand_rotation_net = RotationNet()

    hand_trans_backbone = ResNetBackbone(cfg.trans_resnet_type)
    hand_trans_net = TransNet(hand_trans_backbone)

    if mode == 'train':
        body_backbone.init_weights()
        body_box_net.apply(init_weights)

        hand_roi_net.apply(init_weights)
        hand_backbone.init_weights()
        hand_position_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)

        hand_trans_net.apply(init_weights)
        hand_trans_backbone.init_weights()

    model = Model(body_backbone, body_box_net, hand_roi_net, hand_position_net, hand_rotation_net, hand_trans_net)
    return model
