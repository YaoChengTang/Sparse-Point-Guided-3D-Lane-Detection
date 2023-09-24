# ==============================================================================
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype
from utils.gpu_mem_track import MemTracker
from utils.utils import *
from models.networks.feature_extractor import *
from models.networks import Lane2D, Lane3D
from models.networks.libs.layers import *
from models.networks.PE import PositionEmbeddingLearned
from models.networks.Layers import EncoderLayer
from models.networks.Unet_parts import Down, Up
from models.networks.implicit_module import *

from data.Load_Data import compute_3d_lanes_all_category
from utils.utils import projective_transformation



# overall network
class ImplicitPersFormer(nn.Module):
    def __init__(self, args, anchor_y_steps, anchor_x_steps, x_off_std, z_std):
        super(ImplicitPersFormer, self).__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        self.num_proj = args.num_proj
        self.num_att = args.num_att

        self.num_samples_list         = args.num_samples_list
        self.lane2d_sample_thold_list = args.lane2d_sample_thold_list
        self.lane3d_sample_thold_list = args.lane3d_sample_thold_list
        self.lane2d_seg_num_block     = args.lane2d_seg_num_block
        self.lane2d_offset_num_block  = args.lane2d_offset_num_block
        self.lane3d_seg_num_block     = args.lane3d_seg_num_block
        self.lane3d_offset_num_block  = args.lane3d_offset_num_block

        self.model_name     = args.model_name

        self.N_y       = len(anchor_y_steps)
        anchor_y_steps = torch.from_numpy(anchor_y_steps.astype(np.float32)).cuda()              # (N_y,)
        self.register_buffer('anchor_y_steps', anchor_y_steps)
        anchor_x_steps = torch.from_numpy(anchor_x_steps.astype(np.float32)).cuda()              # (N_anchor, N_y)
        self.register_buffer('anchor_x_steps', anchor_x_steps)

        self.x_off_std = x_off_std
        self.z_std     = z_std
        
        self.sampling_step = args.sampling_step
        self.sampling_size = args.sampling_size

        # define required transformation matrices
        self.M_inv, self.cam_height, self.cam_pitch = self.get_transform_matrices(args)
        if not self.no_cuda:
            self.M_inv = self.M_inv.cuda()

        # Define network
        # backbone: feature_extractor
        self.encoder = self.get_encoder(args)
        self.neck = nn.Sequential(*make_one_layer(self.encoder.dimList[0], args.feature_channels, batch_norm=True),
                                  *make_one_layer(args.feature_channels, args.feature_channels, batch_norm=True))
        
        # 2d lane detector
        self.shared_encoder = Lane2D.FrontViewPathway(args.feature_channels, args.num_proj)
        stride = 2
        self.laneatt_head = Lane2D.LaneATTHead(stride * pow(2, args.num_proj - 1),
                                               args.feature_channels * pow(2, args.num_proj - 2), # no change in last proj
                                               args.im_anchor_origins,
                                               args.im_anchor_angles,
                                               img_w=args.resize_w//2,
                                               img_h=args.resize_h//2,
                                               S=args.S,
                                               anchor_feat_channels=args.anchor_feat_channels,
                                               num_category=args.num_category)
        # self.lane2d_head = LaneImplicit(4, [128,256,512,512], self.num_category+1, 
        #                                   dilation=12, local_fea_size=16,
        #                                   embedding_size=256, leaky=False, legacy=False,
        #                                   sample_ratio=1, sample_step=1, pre_norm_point=False, use_resnet=True,
        #                                   num_samples_list=self.num_samples_list, pos_dim=1,
        #                                   sample_thold_list=self.lane2d_sample_thold_list,
        #                                   seg_num_block=self.lane2d_seg_num_block, 
        #                                   offset_num_block=self.lane2d_offset_num_block)

        # Perspective Transformer: get better bev feature
        self.pers_tr = PerspectiveTransformer(args,
                                              channels=args.feature_channels, # 128
                                              bev_h=args.ipm_h,  # 208
                                              bev_w=args.ipm_w,  # 128
                                              uv_h=args.resize_h//stride,  # 180
                                              uv_w=args.resize_w//stride,  # 240
                                              M_inv=self.M_inv, 
                                              num_att=self.num_att, 
                                              num_proj=self.num_proj, 
                                              nhead=args.nhead,
                                              npoints=args.npoints,
                                              use_all_fea=False)
        
        # BEV feature extractor
        self.bev_head = BEVHead(args, channels=args.feature_channels)

        # 3d lane detector
        self.lane3d_head_list = nn.ModuleList()
        self.frontview_features_channel_list = [512,512,256,128]
        self.dilation_list = [0,0,0,0]   # not used
        self.embedding_size_list = [0,256,256,256]
        for layer_id in range(4) :
            if layer_id==0 :
                lane3d_head = Lane3D.LanePredictionHead(args.feature_channels * pow(2, self.num_proj - 2),
                                                        self.num_lane_type,
                                                        self.num_y_steps,
                                                        args.num_category,
                                                        args.fmap_mapping_interp_index,
                                                        args.fmap_mapping_interp_weight,
                                                        args.no_3d,
                                                        args.batch_norm,
                                                        args.no_cuda)
            else :
                lane3d_head = LaneImplicit(self.frontview_features_channel_list[layer_id], 
                                            num_classes=1, 
                                            dilation=self.dilation_list[layer_id], 
                                            local_fea_size=256,
                                            embedding_size=256, 
                                            leaky=False, legacy=False,
                                            use_resnet=True,
                                            seg_num_block=self.lane3d_seg_num_block, 
                                            num_y_steps=self.num_y_steps,
                                            num_category=args.num_category)
            self.lane3d_head_list.append(lane3d_head)
        # self.lane3d_head_list = nn.ModuleList(self.lane3d_head_list)
        
        # segmentation head
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(512,512,3,2,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.segment_head = SegmentHead(channels=args.feature_channels)
        
        # uncertainty loss weight
        self.uncertainty_loss = nn.Parameter(torch.tensor([args._3d_vis_loss_weight,
                                                            args._3d_prob_loss_weight,
                                                            args._3d_reg_loss_weight,
                                                            args._2d_vis_loss_weight,
                                                            args._2d_prob_loss_weight,
                                                            args._2d_reg_loss_weight,
                                                            args._seg_loss_weight,
                                                            args._3d_point_occ_loss_weight,
                                                            args._3d_point_vis_loss_weight]), requires_grad=True)
        self._initialize_weights(args)

        # self.gpu_tracker = MemTracker()

    def forward(self, input, _M_inv=None,
                aug_mat=None, 
                H_g2im=None, P_g2im=None, H_crop=None, H_im2ipm=None,
                intrinsic=None, extrinsic=None):
        """
            args:
                input:          (B,3,H,W), images, (B,3,360,480);
                _M_inv:         (B,3,3)
                aug_mat:        (B,3,3), matrix, rotation augmentation;
                H_g2im:         (B,3,3), matrix, homography matrix from ground to image coordinate;
                P_g2im:         (B,3,4), matrxi, projection matrix from ground to image coordinate;
                H_crop:         (B,3,3), matrix, the homography matrix transform original image to cropped and resized image;
                H_im2ipm:       (B,3,3), matrix, homography matrix from image to ipm plane;
                intrinsic:      (B,3,3), intrinsic matrix;
                extrinsic:      (B,4,4), extrinsic matrix;
        """
        # self.gpu_tracker.track()
        out_featList = self.encoder(input)
        # self.gpu_tracker.track()
        neck_out = self.neck(out_featList[0])
        # self.gpu_tracker.track()
        frontview_features = self.shared_encoder(neck_out)
        # self.gpu_tracker.track()
        '''
            frontview_features_0 size: torch.Size([4, 128, 180, 240])
            frontview_features_1 size: torch.Size([4, 256, 90, 120])
            frontview_features_2 size: torch.Size([4, 512, 45, 60])
            frontview_features_3 size: torch.Size([4, 512, 22, 30])
        '''

        laneatt_proposals_list = self.laneatt_head(F.avg_pool2d(frontview_features[-1],4,2,1))
        # lane2d_result = self.lane2d_head(frontview_features, sampled_points=points_coord_2d)
        # lane2d_result, gloabl_fea_2d, embedding_2d, local_fea_2d, lane_points_embedding_2d, lane_points_local_fea_2d, lane_points_coord_2d = self.lane2d_head(frontview_features, sampled_points=points_coord_2d)
        # self.gpu_tracker.track()

        projs = self.pers_tr(input, frontview_features, _M_inv)
        # self.gpu_tracker.track()
        '''
            # projs_0 size: torch.Size([4, 128, 208, 128])
            # projs_1 size: torch.Size([4, 256, 104, 64])
            # projs_2 size: torch.Size([4, 512, 52, 32])
            projs_3 size: torch.Size([4, 512, 26, 16])
        '''

        lane3d_result_list = []
        point3d_prob_vis_list = []
        points_3d_info_list = []
        for layer_idx in range(4) :
            # !!! note the module sequence is opposite to feature list, we need to flip feature_list
            # we keep layer_idx as the index of module sequence
            if layer_idx==0 :
                # coarse lane detection
                lane3d_result = self.lane3d_head_list[layer_idx]( projs[-1] )
                lane3d_result_list.append(lane3d_result)
                point3d_prob_vis_list.append(None)
                points_3d_info_list.append(None)

            else :
                coarse_lane3d_result = lane3d_result.detach()
                # sampling points with 3D coor and visible, (N_sampling, B, N_ancho, N_y, 4)
                # sampling points with 2D coordinate (N_sampling, B, N_ancho, N_y, 2)
                # sample_point_info: (size[0]*size[1], B, N_anchor, 3*N_y+N_c)
                points_3d_info, points_2d_coord, sample_x_offset, sample_z, sample_vis, category = self.sampling_points(coarse_lane3d_result=coarse_lane3d_result,
                                                                                        step=self.sampling_step/(2**layer_idx), 
                                                                                        size=self.sampling_size,
                                                                                        aug_mat=aug_mat,
                                                                                        H_g2im=H_g2im, P_g2im=P_g2im, H_crop=H_crop, H_im2ipm=H_im2ipm,
                                                                                        intrinsic=intrinsic, extrinsic=extrinsic)
                
                # fine-level point refinement, (B, 1+1, N_sampling, N_anchor, N_y), if valid and if visible
                # input image resolution is (360,480),
                # the following feature map size is (22, 30), (45, 60), (90, 120), (180, 240)
                # the corresponding scale is           16,        8,        4,         2
                scale = 2**(4-layer_idx)
                # print("-"*10, frontview_features[4-1-layer_idx].shape, points_3d_info.shape, M.shape, scale)
                lane3d_result = self.lane3d_head_list[layer_idx](frontview_features[4-1-layer_idx], 
                                                                    sampled_3d_points=points_3d_info, 
                                                                    points_2d_coord=points_2d_coord,
                                                                    sample_x_offset=sample_x_offset, 
                                                                    sample_z=sample_z,
                                                                    sample_vis=sample_vis, 
                                                                    category=category,
                                                                    scale=scale,
                                                                    coarse_lane3d_result=coarse_lane3d_result, )
                lane3d_result_list.append(lane3d_result)
                # self.gpu_tracker.track()

        cam_height = self.cam_height.to(input.device)
        cam_pitch = self.cam_pitch.to(input.device)
        # self.gpu_tracker.track()

        pre_seg = self.de_conv(projs[3])
        pred_seg_bev_map = self.segment_head(pre_seg)

        # seperate loss weight
        uncertainty_loss = torch.tensor(1.0).to(input.device) * self.uncertainty_loss.to(input.device)

        return laneatt_proposals_list, lane3d_result_list, cam_height, cam_pitch, pred_seg_bev_map, uncertainty_loss
        # return lane2d_result, lane3d_result, cam_height, cam_pitch, uncertainty_loss, \
        #         frontview_features, \
        #        gloabl_fea_2d, embedding_2d, local_fea_2d, lane_points_embedding_2d, lane_points_local_fea_2d, lane_points_coord_2d, \
        #        gloabl_fea_3d, embedding_3d, local_fea_3d, lane_points_embedding_3d, lane_points_local_fea_3d, lane_points_coord_3d

    def _initialize_weights(self, args):
        define_init_weights(self.neck, args.weight_init)
        define_init_weights(self.shared_encoder, args.weight_init)
        define_init_weights(self.laneatt_head, args.weight_init)
        define_init_weights(self.pers_tr, args.weight_init)
        # define_init_weights(self.bev_head, args.weight_init)
        for lane3d_head in self.lane3d_head_list :
            define_init_weights(lane3d_head, args.weight_init)
        define_init_weights(self.segment_head, args.weight_init)
    
    def sampling_points(self, coarse_lane3d_result, step=(1,1), size=(3,3),
                        aug_mat=None, 
                        H_g2im=None, P_g2im=None, H_crop=None, H_im2ipm=None,
                        intrinsic=None, extrinsic=None):
        """
            args:
                coarse_lane3d_result: (B, N_anchor, N_y*3 + N_C);
                step:                 tuple, sampling step size;
                size:                 tuple, sampling number;
                aug_mat:              (B,3,3), matrix, rotation augmentation;
                H_g2im:               (B,3,3), matrix, homography matrix from ground to image coordinate;
                P_g2im:               (B,3,4), matrxi, projection matrix from ground to image coordinate;
                H_crop:               (B,3,3), matrix, the homography matrix transform original image to cropped and resized image;
                H_im2ipm:             (B,3,3), matrix, homography matrix from image to ipm plane;
                intrinsic:            (B,3,3), intrinsic matrix;
                extrinsic:            (B,4,4), extrinsic matrix;
            return:
                sample_point_info:  (size[0]*size[1], B, N_ancho, N_y, 4);
                points_2d_coord:    (size[0]*size[1], B, N_ancho, N_y, 2);
        """
        with torch.no_grad() :
            # lane3d_result = nms_bev(lane3d_result, args)

            # get x,y,z and visible property
            B,N_anchor,N_property = coarse_lane3d_result.shape
            N_y = self.N_y
            x_offset = coarse_lane3d_result[..., :N_y]           # (B, N_anchor, N_y)
            z        = coarse_lane3d_result[..., N_y:2*N_y]      # (B, N_anchor, N_y)
            visible  = coarse_lane3d_result[..., 2*N_y:3*N_y]    # (B, N_anchor, N_y)
            category = coarse_lane3d_result[..., 3*N_y:]         # (B, N_anchor, N_c)
            
            # compute shift for sampling
            x_shift = np.linspace(-(size[0]//2)*step[0], size[0]//2*step[0], size[0], endpoint=True)  # size[0]
            z_shift = np.linspace(-(size[1]//2)*step[1], size[1]//2*step[1], size[1], endpoint=True)  # size[1]
            x_shift = torch.from_numpy(x_shift.astype(np.float32)).to(coarse_lane3d_result.device)
            z_shift = torch.from_numpy(z_shift.astype(np.float32)).to(coarse_lane3d_result.device)
            x_shift, z_shift = torch.meshgrid([x_shift, z_shift])                                   # (size[0], size[1])
            x_shift = x_shift.flatten().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                   # (size[0]*size[1], 1,1,1)
            z_shift = z_shift.flatten().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                   # (size[0]*size[1], 1,1,1)

            # acquire points via shift for x and z, keep y and visible the same
            sample_x_offset = x_offset + x_shift                                             # (size[0]*size[1], B, N_ancho, N_y)
            sample_z        = z + z_shift                                                    # (size[0]*size[1], B, N_ancho, N_y)
            sample_vis      = visible.unsqueeze(0).repeat_interleave(size[0]*size[1], dim=0) # (size[0]*size[1], B, N_ancho, N_y)
            category        = category.unsqueeze(0).repeat_interleave(size[0]*size[1], dim=0)
            # sample_point_info = torch.cat([sample_x_offset,
            #                                sample_z,
            #                                sample_vis,
            #                                category], dim=-1)                                # (size[0]*size[1], B, N_anchor, 3*N_y+N_c)
            # sample_point_info = sample_point_info.to(device=coarse_points.device,
            #                                         dtype=coarse_points.dtype)

            # points_3d_coord:    (N_sampling*B,N_anchor,N_y,3);
            # points_2d_coord:    (N_sampling*B,N_anchor,N_y,2);
            points_3d_coord, points_2d_coord = lane_anchor_project_to_image(sample_x_offset.flatten(0,1), 
                                                                            sample_z.flatten(0,1), 
                                                                            self.anchor_y_steps, self.anchor_x_steps,
                                                                            self.x_off_std, self.z_std, aug_mat, 
                                                                            H_g2im, P_g2im, H_crop, H_im2ipm,
                                                                            intrinsic=intrinsic, extrinsic=extrinsic,
                                                                            sampling_size=size[0]*size[1], )
            points_2d_coord = points_2d_coord.reshape((size[0]*size[1], B, N_anchor, N_y, 2))  # (size[0]*size[1], B, N_anchor, N_y, 2)
            points_3d_coord = points_3d_coord.reshape((size[0]*size[1], B, N_anchor, N_y, 3))  # (size[0]*size[1], B, N_anchor, N_y, 3)
            sample_point_info = torch.cat([points_3d_coord, sample_vis.unsqueeze(-1)], dim=-1) # (size[0]*size[1], B, N_anchor, N_y, 4)

        return sample_point_info, points_2d_coord, \
               sample_x_offset, sample_z, sample_vis, category

    def get_unique_result(self, point3d_prob_vis, points_3d, lane3d_result):
        """
            args:
                point3d_prob_vis: (B, 1+1, N_sampling, N_anchor, N_y), if valid and if visible;
                points_3d:        (N_sampling, B, N_anchor, N_y, 4), coor (3) and visible (1);
                lane3d_result:    (B, N_anchor, 3*N_y+N_c)
        """
        points_prob = point3d_prob_vis[:,0]     # (B, N_sampling, N_anchor, N_y)
        points_vis  = point3d_prob_vis[:,1]     # (B, N_sampling, N_anchor, N_y)

        # find best samplig points according to the probability
        N_sampling,B,N_anchor,N_y,pos_dim = points_3d.shape
        best_idx = torch.argmax(points_prob, dim=1, keepdim=True)    # (B, 1, N_anchor, N_y)

        # obatin x_offset,z,vis of best sampling point
        points_3d  = points_3d.transpose(0,1)                        # (B, N_sampling, N_anchor, N_y, 4)
        points_x   = torch.gather(points_3d[...,0], 1, best_idx)     # (B, 1, N_anchor, N_y)
        points_z   = torch.gather(points_3d[...,2], 1, best_idx)     # (B, 1, N_anchor, N_y)
        points_vis = torch.gather(points_vis, 1, best_idx)           # (B, 1, N_anchor, N_y)
        x_offset       = points_x - self.anchor_x_steps              # (B, 1, N_anchor, N_y)
        # obtain lane line category
        category       = lane3d_result[..., 3*N_y:]                  # (B, N_anchor, N_c)
        # build laneatt format result
        lane3d_result = torch.cat([x_offset,points_z,points_vis,category.unsqueeze(1)], dim=-1).squeeze(1) # (B, N_anchor, 3*N_y+N_c)
        # print("+"*10, lane3d_result.dtype)

        return lane3d_result

    def get_transform_matrices(self, args):
        # define homographic transformation between image and ipm
        org_img_size = np.array([args.org_h, args.org_w])
        resize_img_size = np.array([args.resize_h, args.resize_w])
        cam_pitch = np.pi / 180 * args.pitch

        # image scale matrix
        S_im = torch.from_numpy(np.array([[args.resize_w,              0, 0],
                                          [            0,  args.resize_h, 0],
                                          [            0,              0, 1]], dtype=np.float32))
        S_im_inv = torch.from_numpy(np.array([[1/np.float(args.resize_w),                         0, 0],
                                              [                        0, 1/np.float(args.resize_h), 0],
                                              [                        0,                         0, 1]], dtype=np.float32))
        S_im_inv_batch = S_im_inv.unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # image transform matrix
        H_c = homography_crop_resize(org_img_size, args.crop_y, resize_img_size)
        H_c = torch.from_numpy(H_c).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # camera intrinsic matrix
        K = torch.from_numpy(args.K).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # homograph ground to camera
        H_g2cam = np.array([[1,                             0,               0],
                            [0, np.sin(-cam_pitch), args.cam_height],
                            [0, np.cos(-cam_pitch),               0]])
        H_g2cam = torch.from_numpy(H_g2cam).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # transform from ipm normalized coordinates to ground coordinates
        H_ipmnorm2g = homography_ipmnorm2g(args.top_view_region)
        H_ipmnorm2g = torch.from_numpy(H_ipmnorm2g).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # compute the tranformation from ipm norm coords to image norm coords
        M_ipm2im = torch.bmm(H_g2cam, H_ipmnorm2g)
        M_ipm2im = torch.bmm(K, M_ipm2im)
        M_ipm2im = torch.bmm(H_c, M_ipm2im)
        M_ipm2im = torch.bmm(S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(M_ipm2im,  M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]))
        M_inv = M_ipm2im

        cam_height = torch.tensor(args.cam_height).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)
        cam_pitch = torch.tensor(cam_pitch).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)

        return M_inv, cam_height, cam_pitch

    def get_encoder(self, args):
        if args.encoder == 'ResNext101':
            return deepFeatureExtractor_ResNext101(lv6=False)
        elif args.encoder == 'VGG19':
            return deepFeatureExtractor_VGG19(lv6=False)
        elif args.encoder == 'DenseNet161':
            return deepFeatureExtractor_DenseNet161(lv6=False)
        elif args.encoder == 'InceptionV3':
            return deepFeatureExtractor_InceptionV3(lv6=False)
        elif args.encoder == 'MobileNetV2':
            return deepFeatureExtractor_MobileNetV2(lv6=False)
        elif args.encoder == 'ResNet101':
            return deepFeatureExtractor_ResNet101(lv6=False)
        elif 'EfficientNet' in args.encoder:
            return deepFeatureExtractor_EfficientNet(args.encoder, lv6=False, lv5=False, lv4=False, lv3=False)
        else:
            raise Exception("encoder model in args is not supported")


class PerspectiveTransformer(nn.Module):
    def __init__(self, args, channels, bev_h, bev_w, uv_h, uv_w, M_inv, num_att, num_proj, nhead, npoints, use_all_fea):
        super(PerspectiveTransformer, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.uv_h = uv_h
        self.uv_w = uv_w
        self.M_inv = M_inv
        self.num_att = num_att
        self.num_proj = num_proj
        self.nhead = nhead
        self.npoints = npoints
        
        self.use_all_fea = use_all_fea

        self.query_embeds = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.el = nn.ModuleList()
        self.project_layers = nn.ModuleList()
        self.ref_2d = []
        self.input_spatial_shapes = []
        self.input_level_start_index = []

        uv_feat_c = channels
        for i in range(self.num_proj):
            if i > 0:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
                uv_h = uv_h // 2
                uv_w = uv_w // 2
                if i != self.num_proj-1:
                    uv_feat_c = uv_feat_c * 2
            
            if not self.use_all_fea and i<self.num_proj-1 :
                self.query_embeds.append(None)
                self.pe.append(None)
                self.ref_2d.append(None)
                self.project_layers.append(None)
                self.input_spatial_shapes.append(None)
                self.input_level_start_index.append(None)
                for j in range(self.num_att):
                    self.el.append(None)
                continue

            bev_feat_len = bev_h * bev_w
            query_embed = nn.Embedding(bev_feat_len, uv_feat_c)
            self.query_embeds.append(query_embed)
            position_embed = PositionEmbeddingLearned(bev_h, bev_w, num_pos_feats=uv_feat_c//2)
            self.pe.append(position_embed)

            ref_point = self.get_reference_points(H=bev_h, W=bev_w, dim='2d', bs=1)
            self.ref_2d.append(ref_point)

            size_top = torch.Size([bev_h, bev_w])
            project_layer = Lane3D.RefPntsNoGradGenerator(size_top, self.M_inv, args.no_cuda)
            self.project_layers.append(project_layer)

            spatial_shape = torch.as_tensor([(uv_h, uv_w)], dtype=torch.long)
            self.input_spatial_shapes.append(spatial_shape)

            level_start_index = torch.as_tensor([0.0,], dtype=torch.long)
            self.input_level_start_index.append(level_start_index)

            for j in range(self.num_att):
                encoder_layers = EncoderLayer(d_model=uv_feat_c, dim_ff=uv_feat_c*2, num_levels=1, 
                                              num_points=self.npoints, num_heads=self.nhead)
                self.el.append(encoder_layers)

    def forward(self, input, frontview_features, _M_inv = None):
        projs = []
        for i in range(self.num_proj):
            if i == 0:
                bev_h = self.bev_h
                bev_w = self.bev_w
            else:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
            
            if not self.use_all_fea and i<self.num_proj-1 :
                projs.append(None)
                continue

            bs, c, h, w = frontview_features[i].shape
            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)
            src = frontview_features[i].flatten(2).permute(0, 2, 1)
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=query_embed.device).to(query_embed.dtype)
            bev_pos = self.pe[i](bev_mask).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
            ref_2d = self.ref_2d[i].repeat(bs, 1, 1, 1).to(input.device)
            ref_pnts = self.project_layers[i](_M_inv).unsqueeze(-2)
            input_spatial_shapes = self.input_spatial_shapes[i].to(input.device)
            input_level_start_index = self.input_level_start_index[i].to(input.device)
            for j in range(self.num_att):
                query_embed = self.el[i*self.num_att+j](query=query_embed, value=src, bev_pos=bev_pos, 
                                                        ref_2d = ref_2d, ref_3d=ref_pnts,
                                                        bev_h=bev_h, bev_w=bev_w, 
                                                        spatial_shapes=input_spatial_shapes,
                                                        level_start_index=input_level_start_index)
            query_embed = query_embed.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()
            projs.append(query_embed)
        return projs

    @staticmethod
    def get_reference_points(H, W, Z=8, D=4, dim='3d', bs=1, device='cuda', dtype=torch.long):
        """Get the reference points used in decoder.
        Args:
            H, W spatial shape of bev
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # 2d to 3d reference points, need grid from M_inv
        if dim == '3d':
            raise Exception("get reference poitns 3d not supported")
            zs = torch.linspace(0.5, Z - 0.5, D, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(-1, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(D, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(D, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)

            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H  # ?
            ref_x = ref_x.reshape(-1)[None] / W  # ?
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d   


class BEVHead(nn.Module):
    def __init__(self, args, channels=128):
        super(BEVHead, self).__init__()
        self.size_reduce_layer_1 = Lane3D.SingleTopViewPathway(channels)            # 128 to 128
        self.size_reduce_layer_2 = Lane3D.SingleTopViewPathway(channels*2)          # 256 to 256
        self.size_dim_reduce_layer_3 = Lane3D.EasyDown2TopViewPathway(channels*4)   # 512 to 256

        self.dim_reduce_layers = nn.ModuleList()
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*2,     # 256
                                                        channels,                   # 128
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*4,     # 512
                                                        channels*2,                 # 256
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*4,     # 512
                                                        channels*2,                 # 256
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))

    def forward(self, projs):
        '''
            projs_0 size: torch.Size([4, 128, 208, 128])
            projs_1 size: torch.Size([4, 256, 104, 64])
            projs_2 size: torch.Size([4, 512, 52, 32])
            projs_3 size: torch.Size([4, 512, 26, 16])

            bev_feat_1 size: torch.Size([4, 128, 104, 64])
            bev_feat_2 size: torch.Size([4, 256, 52, 32])
            bev_feat_3 size: torch.Size([4, 256, 26, 16])

            bev_feat   size: torch.Size([4, 512, 26, 16])
        '''
        bev_feat_1 = self.size_reduce_layer_1(projs[0])          # 128 -> 128
        rts_proj_feat_1 = self.dim_reduce_layers[0](projs[1])    # 256 -> 128
        bev_feat_2 = self.size_reduce_layer_2(torch.cat((bev_feat_1, rts_proj_feat_1), 1))     # 128+128 -> 256
        rts_proj_feat_2 = self.dim_reduce_layers[1](projs[2])    # 512 -> 256
        bev_feat_3 = self.size_dim_reduce_layer_3(torch.cat((bev_feat_2, rts_proj_feat_2), 1)) # 256+256 -> 256
        rts_proj_feat_3 = self.dim_reduce_layers[2](projs[3])    # 512 -> 256
        bev_feat = torch.cat((bev_feat_3, rts_proj_feat_3), 1)   # 256+256=512
        return bev_feat


class SegmentHead(nn.Module):
    def __init__(self, channels=128):
        super(SegmentHead, self).__init__()
        self.down1 = Down(channels, channels*2)     # Down(128, 256)
        self.down2 = Down(channels*2, channels*4)   # Down(256, 512)
        self.down3 = Down(channels*4, channels*4)   # Down(512, 512)
        self.up1 = Up(channels*8, channels*2)       # Up(1024, 256)
        self.up2 = Up(channels*4, channels)         # Up(512, 128)
        self.up3 = Up(channels*2, channels)         # Up(256, 128)
        self.segment_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, input):
        x1 = self.down1(input)                      # 128 -> 256
        x2 = self.down2(x1)                         # 256 -> 512
        x3 = self.down3(x2)                         # 512 -> 512
        x_out = self.up1(x3, x2)                    # 512+512 -> 256
        x_out = self.up2(x_out, x1)                 # 256+256 -> 128
        x_out = self.up3(x_out, input)              # 128+128 ->128
        pred_seg_bev_map = self.segment_head(x_out) # 128 -> 1

        return pred_seg_bev_map




class LaneImplicit(nn.Module):
    """
        args:
            in_channels:            channels of global features;
            num_classes:            number of class, bg (0) + all lane categories (1~21);
            dilation:               dilation used to sample local faetures;
            local_fea_size:         channels of local features;
            embedding_size:         size of embeddings;
            leaky:                  relu or leaky_relu;
            legacy:                 using Coscnv1 (False) or Linear (True);
    """
    def __init__(self, in_channels, num_classes, dilation=12, local_fea_size=16,
                 embedding_size=256, leaky=False, legacy=False, use_resnet=False,
                 seg_num_block=3, num_y_steps=10, num_category=21):
        super(LaneImplicit, self).__init__()
        self.num_classes  = num_classes
        self.property_channel = num_classes   # prob

        self.embedding_size   = embedding_size
        self.local_fea_size   = local_fea_size

        self.global_fea_extactor = Compress(in_channels, 256)

        self.local_fea_extactor = LocalFeaLowDimDynamicFusion(in_channels, local_fea_size, embedding_size, 
                                                              point_dim=4, use_resnet=use_resnet)

        self.fusion = nn.Sequential(
            nn.Conv1d(embedding_size*3,embedding_size,1),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(True),
        )
        self.anchor_fea_extractor = nn.Sequential(
            nn.Conv2d(embedding_size,embedding_size,1),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),

            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
            nn.Conv2d(embedding_size,embedding_size,kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),

            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),padding=(0,0)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),padding=(0,0)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),padding=(0,0)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),padding=(0,0)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
        )
        self.anchor_predictor = nn.Sequential(
            nn.Conv2d(embedding_size+3,embedding_size,kernel_size=(1,3),padding=(0,1)),
            nn.Conv2d(embedding_size,embedding_size//4,1),
            nn.Conv2d(embedding_size//4,3,1),
        )
        self.lane_fea_extractor = nn.Sequential(
            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),stride=(1,2),padding=(0,1)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),padding=(0,0)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
            nn.Conv2d(embedding_size,embedding_size,kernel_size=(1,3),padding=(0,0)),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(True),
        )
        self.lane_predictor = nn.Sequential(
            nn.Conv1d(embedding_size+num_category,embedding_size,1),
            nn.Conv1d(embedding_size,embedding_size//4,1),
            nn.Conv1d(embedding_size//4,num_category,1),
        )

        # self.gpu_tracker = MemTracker()


    def forward(self, frontview_features, 
                sampled_3d_points=None, points_2d_coord=None, 
                scale=None,
                sample_x_offset=None, 
                sample_z=None,
                sample_vis=None, 
                category=None,
                coarse_lane3d_result=None):
        """
            args:
                frontview_features:     (B,C,H,W);
                sampled_3d_points:      (N_sampling, B, N_ancho, N_y, 4), 3d coordinates and visible;
                points_2d_coord:        (N_sampling, B, N_ancho, N_y, 2), 2d coordinates;
                sample_x_offset:        (N_sampling, B, N_anchor, N_y)
                sample_z:               (N_sampling, B, N_anchor, N_y)
                sample_vis:             (N_sampling, B, N_anchor, N_y)
                category:               (N_sampling, B, N_anchor, N_c);
                coarse_lane3d_result:   (B, N_anchor, N_y*3 + N_C);
        """
        # self.gpu_tracker.track()
        # get global features, (B, 256)
        gloabl_fea = self.global_fea_extactor( frontview_features )
        # self.gpu_tracker.track()

        # get points embeddings, (B, 256, N_sampling*N_anchor*N_y)
        # get local features, (B, 256, N_sampling*N_anchor*N_y)
        embedding, local_fea = self.local_fea_extactor( frontview_features, 
                                                        sampled_3d_points, points_2d_coord, 
                                                        scale=scale, )
        # print("!"*30, layer_id, " gloabl_fea:", gloabl_fea.shape, " embedding:", embedding.shape, " local_fea:", local_fea.shape)
        # self.gpu_tracker.track()

        B, channel, N_total = embedding.shape
        gloabl_fea = gloabl_fea.unsqueeze(-1).repeat_interleave(N_total, dim=2)
        fea = torch.cat([gloabl_fea, embedding, local_fea], dim=1)             # (B, 256*3, N_sampling*N_anchor*N_y)
        fea = self.fusion(fea)                                                 # (B, 256, N_sampling*N_anchor*N_y)

        N_sampling, B, N_anchor, N_y, pos_dim = sampled_3d_points.shape
        lane_anchor_ner_fea = fea.reshape((B, 256, N_sampling, N_anchor, N_y))      # (B, 256, N_sampling, N_anchor, N_y)

        anchor_ner_fea = lane_anchor_ner_fea.permute(0,3,1,4,2).reshape(B*N_anchor, 256, N_y, N_sampling) # (B*N_anchor, 256, N_y, N_sampling)
        fine_anchor_fea = self.anchor_fea_extractor(anchor_ner_fea).squeeze(-1)                           # (B*N_anchor, 256, N_y)
        fine_anchor_fea = fine_anchor_fea.reshape(B,N_anchor,256,N_y).transpose(1,2)                      # (B, 256, N_anchor, N_y)
        coarse_anchor_fea = coarse_lane3d_result[:,:,:3*N_y].reshape(B,N_anchor,3,N_y).transpose(1,2)     # (B, 3, N_anchor, N_y)
        anchor_property = self.anchor_predictor( torch.cat([coarse_anchor_fea,fine_anchor_fea], dim=1) )  # (B, 3, N_anchor, N_y)
        anchor_property = anchor_property.transpose(1,2).reshape(B,N_anchor,3*N_y)                        # (B, N_anchor, 3*N_y)

        fine_lane_fea = self.lane_fea_extractor(fine_anchor_fea).squeeze(-1)        # (B, 256, N_anchor)
        coarse_lane_fea = coarse_lane3d_result[:,:,3*N_y:].transpose(1,2)           # (B, N_c, N_anchor)
        lane_property = self.lane_predictor( torch.cat([coarse_lane_fea,fine_lane_fea], dim=1) )   # (B, N_c, N_anchor)
        lane_property = lane_property.transpose(1,2)                                               # (B, N_anchor, N_c)

        # get expectation
        lane_result = torch.cat([anchor_property,lane_property], dim=-1)                            # (B, N_anchor, 3*N_y+N_c)

        return lane_result



class ImplicitSamplingLEResHead(nn.Module):
    def __init__(self, in_channels, num_classes, dilation=12, local_fea_size=16,
                 embedding_size=256, leaky=False, legacy=False, num_block=3, binary_seg=False):
        super(ImplicitSamplingLEResHead, self).__init__()
        self.num_classes = num_classes
        self.binary_seg  = binary_seg
        # self.fc_p = nn.Conv1d(2, embedding_size, 1)

        self.block_list = nn.ModuleList()
        for block_idx in range(num_block) :
            block = CResnetLEResBlockConv1d(in_channels, embedding_size, 
                                            local_fea_c_dim=local_fea_size, legacy=legacy)
            self.block_list.append(block)

        if not leaky :
            self.actvn = lambda x: F.relu(x)
            self.bn    = CBatchNorm1d(in_channels, embedding_size)
        else :
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.bn    = CBatchNorm1d_legacy(in_channels, embedding_size)
        
        self.pos_prob = nn.Sequential(
                            nn.Conv1d(embedding_size, embedding_size//4, 1),
                            nn.LeakyReLU(),
                            nn.BatchNorm1d(embedding_size//4, affine=False),
                            nn.Conv1d(embedding_size//4, embedding_size//16, 1),
                            nn.LeakyReLU(),
                            nn.Conv1d(embedding_size//16, self.num_classes, 1),
        )

        print("using ImplicitSamplingLEResHead: num_classes/{}".format(self.num_classes))
    
    
    def forward(self, x, embedded_points, local_fea=None, z=None):
        """
            args:
                x:               Tensor, (B,C), global feature;
                embedded_points: Tensor, (B,C',N), embeddings;
                local_fea:       Tensor, (B,C'',N), local features;
        """
        net = embedded_points
        # return self.fc_out(self.actvn(net))

        if z is not None:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z
        
        for block in self.block_list :
            net = block(net, x, local_fea)

        # out = self.fc_out(self.actvn(self.bn(net, x)))
        # out = self.fc_out(net)

        out = self.pos_prob(net)

        return out



def lane_anchor_project_to_image(x_offsets, z_g, anchor_y_steps, anchor_x_steps,
                                x_off_std, z_std, aug_mat, 
                                H_g2im, P_g2im, H_crop, H_im2ipm,
                                intrinsic=None, extrinsic=None,
                                sampling_size=None):
    """
        args:
            x_offsets:      (N_sampling*B,N_anchor,N_y)
            z_g:            (N_sampling*B,N_anchor,N_y), predicted/GT anchors;
            anchor_y_steps: (N_y,), anchors in y-dim;
            anchor_x_steps: (N_anchor, N_y), anchor in x-dims;
            x_off_std:      float, x-dim std used to normalize anchor in dataloader;
            z_std:          float, z-dim std used to normalize anchor in dataloader;
            aug_mat:        (B,3,3), matrix, rotation augmentation;
            H_g2im:         (B,3,3), matrix, homography matrix from ground to image coordinate;
            P_g2im:         (B,3,4), matrxi, projection matrix from ground to image coordinate;
            H_crop:         (B,3,3), matrix, the homography matrix transform original image to cropped and resized image;
            H_im2ipm:       (B,3,3), matrix, homography matrix from image to ipm plane;
            intrinsic:      (B,3,3), intrinsic matrix;
            extrinsic:      (B,4,4), extrinsic matrix;
            sampling_size:  int, number of sampled points;
        
        return:
            points_3d_coord:    (N_sampling*B,N_anchor,N_y,3);
            points_2d_coord:    (N_sampling*B,N_anchor,N_y,2);
    """
    B,N_anchor,num_y_steps = x_offsets.shape
    # num_y_steps = len(anchor_y_steps)
    # # recover position (x,z) from normalization
    # x_offsets  = anchor[..., :num_y_steps] * x_off_std                 # (B,N_anchor,N_y)
    # z_g        = anchor[..., num_y_steps:2*num_y_steps] * z_std        # (B,N_anchor,N_y)
    # visibility = anchor[:, 2*num_y_steps:3*num_y_steps]                # (B,N_anchor,N_y)

    x_gflat    = x_offsets + anchor_x_steps                                  # (N_sampling*B,N_anchor,N_y)
    y_gflat    = anchor_y_steps.view((1,1,num_y_steps)).expand_as(x_offsets) # N_y -> (N_sampling*B, N_anchor, N_y)
    
    # transform lane detected in flat ground space to 3d ground space, code is copied from transform_lane_gflat2g
    h_cam = torch.tile(extrinsic, (sampling_size,1,1))[:,2:3,3:4]               # (N_sampling*B,)
    try :
        x_g = x_gflat - x_gflat * z_g / h_cam                               # (N_sampling*B,N_anchor,N_y)
        y_g = y_gflat - y_gflat * z_g / h_cam                               # (N_sampling*B,N_anchor,N_y)
    except Exception as err :
        raise Exception(err, x_gflat.shape, y_gflat.shape, z_g.shape, h_cam.shape, extrinsic.shape, sampling_size)
    points_3d_coord = torch.stack([x_g,y_g,z_g], axis=-1)                   # (N_sampling*B,N_anchor,N_y,3)

    # recover position from augmentation (rotation, crop), flat ground to image
    ## transformation matrix
    # torch.einsum, bmm
    H_g2im = torch.bmm(H_crop, H_g2im)                                # (B,3,3)
    if aug_mat is not None :
        H_g2im = torch.bmm(aug_mat, H_g2im)                           # (B,3,3)
    H_g2im = torch.tile(H_g2im, (sampling_size,1,1))                  # (N_sampling*B,3,3)
    ## homogeneous coordinate
    ones = torch.ones_like(y_gflat, )
    coordinates = torch.stack((x_gflat, y_gflat, ones), dim=1)        # (N_sampling*B,3,N_anchor,N_y)
    ## flat ground to image plane
    trans = torch.bmm(H_g2im, coordinates.view(B,3,-1))               # (N_sampling*B,3,N_anchor*N_y)
    x_2d = trans[:, 0, :] / trans[:, 2, :]                            # (N_sampling*B,N_anchor*N_y)
    y_2d = trans[:, 1, :] / trans[:, 2, :]                            # (N_sampling*B,N_anchor*N_y)
    # z_3d = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
    # x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
    points_2d_coord = torch.stack([x_2d,y_2d], axis=-1)               # (N_sampling*B,N_anchor*N_y,2)
    points_2d_coord = points_2d_coord.view(B,N_anchor,num_y_steps,2)  # (N_sampling*B,N_anchor,N_y,2)

    return points_3d_coord, points_2d_coord