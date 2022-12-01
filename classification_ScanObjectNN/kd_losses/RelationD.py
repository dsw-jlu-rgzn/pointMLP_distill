from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import sys

sys.path.append('..')
import torch.nn.functional as F
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from models.pointmlp import index_points, farthest_point_sample, knn_point
import models as models
from models.pointmlp import ConvBNReLU1D


def cal_cosine_similarity(x):
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise


class RelationCos(nn.Module):
    def __init__(self, w_dist=15, w_angle=30, in_out_channels_s=[256, 1024], in_out_channels_t=[1024, 1024], k=12, sample_point=32):
        super(RelationCos, self).__init__()
        self.GraphConv_s = ConvBNReLU1D(in_channels=in_out_channels_s[0], out_channels=in_out_channels_s[1])
        self.GraphConv_t = ConvBNReLU1D(in_channels=in_out_channels_t[0], out_channels=in_out_channels_t[1])
        self.k = k
        self.sample_point = sample_point

    def GraphOperation(self, grouped_points, GraphConv):
        x = grouped_points
        b, n, s, d = x.size()  # [2, 64, 16, 512]
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = GraphConv(x)
        batch_size, _, _ = x.size()
        x = torch.nn.functional.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1)
        return x

    def forward(self, feature_s, xyz_s, feature_t, xyz_t):
        fps_idx_t = pointnet2_utils.furthest_point_sample(xyz_t, self.sample_point).long()
        new_xyz_t = index_points(xyz_t, fps_idx_t)
        # new_points_t = index_points(feature_t, fps_idx_t)
        idx_t = knn_point(self.k, xyz_t, new_xyz_t)

        # grouped_xyz_t = index_points(xyz_t, idx_t)  # [B, npoint, k, 3]
        grouped_points_t = index_points(feature_t, idx_t)  # [2, 64, 16, 128]

        idx_s = knn_point(self.k, xyz_s, new_xyz_t)
        # grouped_xyz_s = index_points(xyz_s, idx_s)  # [2, 64, 16, 3]
        grouped_points_s = index_points(feature_s, idx_s)  # [2, 64, 16, 512]

        out_point_s = self.GraphOperation(grouped_points_s, self.GraphConv_s)
        out_point_t = self.GraphOperation(grouped_points_t, self.GraphConv_t)  # B*n*feature
        # loss = 0
        # for i in range(out_point_s.shape[0]):
        #     loss_ = self.w_dist * self.rkd_dist(out_point_s[i], out_point_t[i].detach()) + self.w_angle * self.rkd_angle(out_point_s[i], out_point_t[i].detach())
        #     loss = loss + loss_
        # sim_s = cal_cosine_similarity(out_point_s)
        # sim_t = cal_cosine_similarity(out_point_t)
        # loss = F.smooth_l1_loss(sim_s, sim_t)
        return out_point_s, out_point_t


class RKD(nn.Module):

    def __init__(self, w_dist=10, w_angle=20, p=2, add_weight=False):#
        super(RKD, self).__init__()

        self.w_dist = w_dist
        self.w_angle = w_angle
        self.p = p
        self.add_weight = add_weight

    def attention_map(self, fm, eps=1e-6):  # 2*64*1024 ->2*1024*64
        am = fm.transpose(1, 2)
        am = torch.pow(torch.abs(am), self.p)
        am, _ = torch.max(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=2, keepdim=True)
        am = torch.div(am, norm + eps)

        # am = torch.pow(torch.abs(fm), self.p)
        # am = torch.sum(am, dim=1, keepdim=True)
        # norm = torch.norm(am, dim=(2, 3), keepdim=True)
        # am = torch.div(am, norm + eps)
        return am

    def forward(self, feat_s, feat_t):
        # loss = 0
        # for i in range(feat_s.shape[0]):
        #     loss_ = self.w_dist * self.rkd_dist(feat_s[i], feat_t[i]) + \
        #        self.w_angle * self.rkd_angle(feat_s[i], feat_t[i])
        #     loss = loss + loss_
        # sim_s = cal_cosine_similarity(feat_s)
        # sim_t = cal_cosine_similarity(feat_t)
        # loss = F.smooth_l1_loss(sim_s, sim_t)#
        # loss = F.smooth_l1_loss(feat_s, feat_t)
        # loss = F.mse_loss(self.attention_map(feat_s), self.attention_map(feat_t))
        if self.add_weight:
            weight, _ = torch.max(feat_t, 1)  # 2*512
            weight_soft = torch.nn.functional.softmax(weight, 1).unsqueeze(1)  # 2*512
            feat_s = feat_s * weight_soft * weight_soft.shape[2]
            feat_t = feat_t * weight_soft * weight_soft.shape[2]
            # loss = F.mse_loss(fm_s, fm_t)
        loss = F.mse_loss(feat_s, feat_t)
        return loss





class LocalRegionMulti(nn.Module):
    def __init__(self, in_out_channels_s=None,
                 in_out_channels_t=None, k=12, groups=64):
        super(LocalRegionMulti, self).__init__()
        if in_out_channels_t is None:
            in_out_channels_t = [[1024, 1024], [256, 1024], [256, 1024], [256, 1024]]
        if in_out_channels_s is None:
            in_out_channels_s = [[256, 1024], [256, 1024], [256, 1024], [256, 1024]]
        self.GraphConv_t_list = []
        self.GraphConv_s_list = []
        for i in range(len(in_out_channels_t)):
            graph_conv_s = ConvBNReLU1D(in_channels=in_out_channels_s[i][0], out_channels=in_out_channels_s[i][1])
            graph_conv_t = ConvBNReLU1D(in_channels=in_out_channels_t[i][0], out_channels=in_out_channels_t[i][1])
            self.GraphConv_s_list.append(graph_conv_s)
            self.GraphConv_t_list.append(graph_conv_t)
        self.GraphConv_t_list = nn.ModuleList(self.GraphConv_t_list)
        self.GraphConv_s_list = nn.ModuleList(self.GraphConv_s_list)
        self.k = k
        self.groups = groups


    def GraphOperation(self, grouped_points, graph_conv):
        x = grouped_points
        b, n, s, d = x.size()  # [2, 64, 16, 512]
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = graph_conv(x)
        batch_size, _, _ = x.size()
        x = torch.nn.functional.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1)
        return x


    def forward(self, feature_s_list, xyz_s_list, feature_t_list, xyz_t_list):
        fps_idx_t = pointnet2_utils.furthest_point_sample(xyz_t_list[-1], self.groups).long()
        new_xyz_t = index_points(xyz_t_list[-1], fps_idx_t) # 2, 64, 3
        # new_points_t = index_points(feature_t, fps_idx_t)
        # idx_t = knn_point(self.k, xyz_t, new_xyz_t)
        # grouped_xyz_t = index_points(xyz_t, idx_t)  # [B, npoint, k, 3]
        # grouped_points_t = index_points(feature_t, idx_t)  # [2, 64, 16, 128]

        # idx_s = knn_point(self.k, xyz_s, new_xyz_t)
        # grouped_xyz_s = index_points(xyz_s, idx_s)  # [2, 64, 16, 3]
        # grouped_points_s = index_points(feature_s, idx_s)  # [2, 64, 16, 512]
        # every procedure share the same region, each region is represent as grouped_points.
        # new_xyz_t is the center of each region, so we need calculate idx_t and idx_s through new_xyz_t and k.
        out_point_s_list = []
        out_point_t_list = []
        for i in range(4):
            idx_t = knn_point(self.k, xyz_t_list[i], new_xyz_t)
            grouped_points_t = index_points(feature_t_list[i], idx_t)
            idx_s = knn_point(self.k, xyz_s_list[i], new_xyz_t)
            grouped_points_s = index_points(feature_s_list[i], idx_s)
            out_point_s = self.GraphOperation(grouped_points_s, self.GraphConv_s_list[i])
            out_point_t = self.GraphOperation(grouped_points_t, self.GraphConv_t_list[i])  # B*n*feature
            out_point_s_list.append(out_point_s)
            out_point_t_list.append(out_point_t)
        return out_point_s_list, out_point_t_list
