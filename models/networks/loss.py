import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


def implicit_loss_func(gt_pitch, pred_pitch,  
                        gt_hcam, pred_hcam,
                        seg_bev_map, pred_seg_bev_map, 
                        gt_lane2d, lane2d_result,  
                        gt_lane3d, lane3d_result_list,
                        args=None, 
                        loss_line2d_func=None, 
                        loss_line3d_func=None,
                        N_y=None,
                        anchor_x_steps=None,
                        step=None) :
    """
        args:
            gt_pitch, pred_pitch,  gt_hcam, pred_hcam;

            seg_bev_map:            (B,H,W), GT BEV binary seg map;
            pred_seg_bev_map:       (B,H,W), predicted BEV binary seg map;

            gt_lane2d:              (B, N_lane2d, num_category + 1 + 1 + 2*n_offsets);
            lane2d_result:          (B, N_lane2d, num_category + 1 + 1 + 2*n_offsets);

            gt_lane3d:              (B, N_anchor, 3*N_y+N_c);
            lane3d_result_list:     [(B, N_anchor, 3*N_y+N_c), ...], length is 4;
            
            point3d_prob_vis_list:  [(B, 1+1, N_sampling, N_anchor, N_y), ], length is 3;
            points_3d_info_list:    [(N_sampling, B, N_ancho, N_y, 4), ]

            N_y:                    float;
            anchor_x_steps:         (N_anchor, N_y);
            step:                   float;
    """
    # pitch_loss  = torch.sum(torch.abs(gt_pitch-pred_pitch))
    # hcam_loss   = torch.sum(torch.abs(gt_hcam-pred_hcam))

    # segmentation loss
    loss_seg = F.binary_cross_entropy_with_logits(pred_seg_bev_map, seg_bev_map)
    
    # 2d lane line loss
    loss_att, loss_att_dict = loss_line2d_func(lane2d_result, gt_lane2d,
                                                cls_loss_weight=args.cls_loss_weight,
                                                reg_vis_loss_weight=args.reg_vis_loss_weight)
    
    # 3d lane line loss
    for idx, lane3d_result in enumerate(lane3d_result_list) :
        tmp_loss_3d, tmp_loss_3d_dict = loss_line3d_func(lane3d_result, gt_lane3d, pred_hcam, gt_hcam, pred_pitch, gt_pitch, idx=idx)
        if idx==0 :
            loss_3d, loss_3d_dict = tmp_loss_3d, tmp_loss_3d_dict
        else :
            loss_3d += tmp_loss_3d
            for key in loss_3d_dict :
                loss_3d_dict[key] += tmp_loss_3d_dict[key]

    return loss_seg, \
           loss_att, loss_att_dict, \
           loss_3d, loss_3d_dict


