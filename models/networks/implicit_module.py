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

from utils.gpu_mem_track import MemTracker

from IPython import embed

#__all__ = ['mish','Mish']



class Compress(nn.Module):
    # Reduce channel (typically to 128), RESA code use no BN nor ReLU

    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(Compress, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, reduce)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return x


class LocalFeaLowDimDynamicFusion(nn.Module):
    # sample local features according to point coordinates,
    # then fuse the local features and the point coodinates in low-dimension.
    # points coordinate: (idx_h, idx_W)

    def __init__(self, in_channel=16, out_local_fea_channel=16, out_embed_channel=256, 
                 point_dim=3, use_resnet=False):
        super(LocalFeaLowDimDynamicFusion, self).__init__()
        self.in_channel = in_channel
        self.use_resnet = use_resnet
        self.point_dim  = point_dim
        
        self.local_fea_conv = nn.Conv1d(in_channel, out_local_fea_channel, 1)
        self.embedding_conv = nn.Conv1d(point_dim, out_embed_channel, 1)

        # self.gpu_tracker = MemTracker()
        print("using LocalFeaLowDimDynamicFusion: point_dim:{}".format(point_dim))

    def forward(self, fea, sampled_3d_points, points_2d_coord, scale=None):
        """
        args:
            fea:               (B, C, H, W), fature at specific scale;
            sampled_3d_points: (N_sampling, B, N_anchor, N_y, 4), 3d coord, visible;
            points_2d_coord:   (N_sampling, B, N_anchor, N_y, 2), 2d coord;
            scale:             int, downsampling scale from highest resolution t current resolution;
        """
        # rescale 2D point coordinate
        ner_coordinates = points_2d_coord / scale
        # rescale to [-1,1] for following grid sampling
        B,C,H,W = fea.shape
        norm_weight = torch.tensor([1/W, 1/H], device=points_2d_coord.device)
        ner_coordinates = ner_coordinates * norm_weight * 2 - 1
        N_s, B, N_anchor, N_y, pos_dim = ner_coordinates.shape
        size = N_s * B * N_anchor * N_y * pos_dim
        # print("-"*10, "scale:", scale, "   H,W:", H, W, "   ner_coordinates: >1: {}/{}, <-1: {}/{}".format( (ner_coordinates>1).sum(),
        #                                                                (ner_coordinates>1).sum() / size * 100,
        #                                                                (ner_coordinates<-1).sum(),
        #                                                                (ner_coordinates<-1).sum() / size * 100, ))
        # self.gpu_tracker.track()

        # get local features
        N_sampling, B, N_anchor, N_y, pos_dim = ner_coordinates.shape
        ner_coordinates = ner_coordinates.transpose(0,1)                                  # (B, N_sampling, N_ancho, N_y, 2)
        ner_coordinates = ner_coordinates.reshape((B, N_sampling*N_anchor, N_y, pos_dim)) # (B, N_sampling*N_anchor, N_y, 2)
        local_fea = F.grid_sample(fea, ner_coordinates)                                   # (B, C, N_sampling*N_anchor, N_y)
        local_fea = local_fea.reshape((B,C,-1))                                           # (B, C, N_sampling*N_anchor*N_y)
        local_fea = self.local_fea_conv(local_fea)                                        # (B, C', N_sampling*N_anchor*N_y)
        # self.gpu_tracker.track()

        # fusion in low-dimension
        sampled_points  = sampled_3d_points.permute(1,4,0,2,3)                    # (B, 4, N_sampling, N_ancho, N_y)
        embedded_points = sampled_points.reshape((B, self.point_dim, -1))         # (B, 4, N_sampling*N_ancho*N_y)
        embedded_points = self.embedding_conv(embedded_points)                    # (B, C'', N_sampling*N_ancho*N_y)
        # self.gpu_tracker.track()

        return embedded_points, local_fea
        

class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetGEBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super(CResnetGEBlockConv1d, self).__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
            self.bn_2 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)
            self.bn_2 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        
        print("CResnetGEBlockConv1d: shortcut: {}".format(self.shortcut))

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            # print("c: {}, x: {}".format(c.shape, x.shape))
            x_s = self.bn_2(c.unsqueeze(-1), x)

        return x_s + dx


class CResnetLEBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, local_fea_c_dim=16, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super(CResnetLEBlockConv1d, self).__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
            self.bn_2 = CBatchNorm1d(
                local_fea_c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)
            self.bn_2 = CBatchNorm1d_legacy(
                local_fea_c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        
        print("CResnetLEBlockConv1d: shortcut: {}".format(self.shortcut))

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c, local_fea):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            # print("c: {}, x: {}".format(c.shape, x.shape))
            x_s = self.bn_2(x, local_fea)

        return x_s + dx


class CResnetLEResBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, local_fea_c_dim=16, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super(CResnetLEResBlockConv1d, self).__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
            self.bn_2 = CBatchNorm1d(
                local_fea_c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)
            self.bn_2 = CBatchNorm1d_legacy(
                local_fea_c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.fc_2 = nn.Conv1d(size_h, size_out, 1)
        self.fusion = nn.Conv1d(size_out*2, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        
        print("CResnetLEResBlockConv1d: shortcut: {}".format(self.shortcut))

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c, local_fea):
        """
            args:
                x:         Tensor, (B,C',N), embeddings;
                c:         Tensor, (B,C), global feature;
                local_fea: Tensor, (B,C'',N), local features;
        """
        # print("->"*15, x.shape, c.shape, local_fea.shape, self.bn_0.c_dim, self.bn_0.f_dim)
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            # print("c: {}, x: {}".format(c.shape, x.shape))
            x_s = x
            dx_local = self.fc_2(self.actvn(self.bn_2(x, local_fea)))
            dx = torch.cat((dx_local, dx), dim=1)
            dx = self.fusion(dx)

        return x_s + dx


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0), "x:{}, c:{}".format(x.shape, c.shape))
        assert(c.size(1) == self.c_dim, "c:{}".format(c.shape))

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta
        # print("x: {}, c: {}, gamma: {}, beta: {}, out: {}".format(x.shape, c.shape, gamma.shape, beta.shape, out.shape))

        return out


class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out
