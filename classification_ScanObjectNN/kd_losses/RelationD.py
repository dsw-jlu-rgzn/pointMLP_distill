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
from models.pointmlp import index_points, farthest_point_sample, knn_point, get_activation
import models as models
from models.pointmlp import ConvBNReLU1D


def cal_cosine_similarity(x):
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise



class SingleClassifier(nn.Module):
    def __init__(self, in_channel=1024,class_num=15, activation="relu" ):
        super(SingleClassifier, self).__init__()
        self.in_channel = in_channel
        self.class_num = class_num
        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

    def forward(self, p):#8, 64, 1024
        p = F.adaptive_max_pool1d(p, 1).squeeze(dim=-1)
        p = self.classifier(p)
        return p
class TeacherFusion(nn.Module):
    def __init__(self, in_channel=2048, output_channel=2, reduction=16):
        super(TeacherFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, output_channel, bias=False),
            nn.Softmax()
        )

        # for module in self.fc:
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                #torch.nn.init.constant_(module.bias, 0)

    def forward(self, p1, p2): #B*n*feature
        p = torch.cat([p1, p2], dim=-1)#B*n*f
        b, n, c = p.size()
        p = p.transpose(1,2)#b,c,n
        y = self.avg_pool(p).view(b, c)#b,c
        #  fr_features.append(fusion_score[:,0].unsqueeze(-1)*hr_feature+fusion_score[:,1].unsqueeze(-1)*lr_feature)
        y = self.fc(y).view(b, -1, 1)#b*2*1  b*c*n
        new_p = y[:, 0].unsqueeze(-1) * p1 + y[:, 1].unsqueeze(-1) * p2
        return new_p


class MultiTeacherOperation(nn.Module):
    def __init__(self, in_channels=[256, 1024, 1024], out_channels=[1024, 1024, 1024], k=12, sample_point=32):
        super(MultiTeacherOperation, self).__init__()
        self.GraphConv_s = ConvBNReLU1D(in_channels=in_channels[0], out_channels=out_channels[0])
        self.GraphConv_t1 = ConvBNReLU1D(in_channels=in_channels[1], out_channels=out_channels[1])
        self.GraphConv_t2 = ConvBNReLU1D(in_channels=in_channels[2], out_channels=out_channels[2])

        self.GraphConv_tf = ConvBNReLU1D(in_channels=in_channels[2], out_channels=out_channels[2])
        self.k = k
        self.sample_point = sample_point
        self.TeacherFusion = TeacherFusion()
        self.clssifier_fusion = SingleClassifier()
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

    def forward(self, feature_s, xyz_s, feature_t1, xyz_t1, feature_t2, xyz_t2, total_xyz=None):
        if total_xyz is None:
            total_xyz = xyz_t1
        # 1. 从xyz_t中挑选出sample_point个点, 输出其id
        # 2. 将fps中的id转为具体的点位置
        # 3. 求出knn后的点id
        fps_idx_t = pointnet2_utils.furthest_point_sample(total_xyz, self.sample_point).long()
        new_xyz_total = index_points(total_xyz, fps_idx_t)


        idx_t1 = knn_point(self.k, xyz_t1, new_xyz_total)
        grouped_points_t1 = index_points(feature_t1, idx_t1)  # [2, 64, 16, 128]

        idx_t2 = knn_point(self.k, xyz_t2, new_xyz_total)
        grouped_points_t2 = index_points(feature_t2, idx_t2)  # [2, 64, 16, 128]
        # grouped_xyz_t = index_points(xyz_t, idx_t)  # [B, npoint, k, 3]
        # 4. 提取坐标

        idx_s = knn_point(self.k, xyz_s, new_xyz_total)
        # grouped_xyz_s = index_points(xyz_s, idx_s)  # [2, 64, 16, 3]
        grouped_points_s = index_points(feature_s, idx_s)  # [2, 64, 16, 512]

        out_point_s = self.GraphOperation(grouped_points_s, self.GraphConv_s)
        out_point_t1 = self.GraphOperation(grouped_points_t1, self.GraphConv_t1)#2,512,128
        out_point_t2 = self.GraphOperation(grouped_points_t2, self.GraphConv_t2)
        new_feature_fusion = self.TeacherFusion(out_point_t1, out_point_t2)#(2,64,1024)
        cls_output = self.clssifier_fusion(new_feature_fusion.transpose(1, 2)) #(2,15)
        # B*n*feature
        # loss = 0
        # for i in range(out_point_s.shape[0]):
        #     loss_ = self.w_dist * self.rkd_dist(out_point_s[i], out_point_t[i].detach()) + self.w_angle * self.rkd_angle(out_point_s[i], out_point_t[i].detach())
        #     loss = loss + loss_
        # sim_s = cal_cosine_similarity(out_point_s)
        # sim_t = cal_cosine_similarity(out_point_t)
        # loss = F.smooth_l1_loss(sim_s, sim_t)
        return out_point_s, new_feature_fusion, cls_output#(2, 64, 1024) (2,64,1024) (2,15)


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

    def forward(self, feature_s, xyz_s, feature_t, xyz_t, total_xyz=None):
        if total_xyz is None:
            total_xyz = xyz_t
        # 1. 从xyz_t中挑选出sample_point个点, 输出其id
        fps_idx_t = pointnet2_utils.furthest_point_sample(total_xyz, self.sample_point).long()
        #2. 将fps中的id转为具体的点位置
        new_xyz_total = index_points(total_xyz, fps_idx_t)
        # new_points_t = index_points(feature_t, fps_idx_t)
        #3. 求出knn后的点id
        idx_t = knn_point(self.k, xyz_t, new_xyz_total)

        # grouped_xyz_t = index_points(xyz_t, idx_t)  # [B, npoint, k, 3]
        # 4. 提取坐标
        grouped_points_t = index_points(feature_t, idx_t)  # [2, 64, 16, 128]

        idx_s = knn_point(self.k, xyz_s, new_xyz_total)
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


if __name__ == '__main__':
    teacherfusion = TeacherFusion().cuda()
    cls = SingleClassifier().cuda()
    p1 = torch.randn(8, 64, 1024).cuda()
    p2 = torch.randn(8, 64, 1024).cuda()
    p3 = teacherfusion(p1, p2)
    p3 = p3.transpose(1, 2)#
    output = cls(p3)
    print(p3.shape)
    feature_s, xyz_s = torch.rand(2, 32, 256),  torch.rand(2, 32, 3)
    feature_t1, xyz_t1 = torch.rand(2, 64, 1024),  torch.rand(2, 64, 3)
    feature_t2, xyz_t2 = torch.rand(2, 32, 1024), torch.rand(2, 32, 3)
    adaptive_xyz = torch.rand(2, 128, 3)
    net_s_trans = MultiTeacherOperation(in_channels=[256, 1024, 1024],
                                        out_channels=[1024, 1024, 1024],
                                        k=16,
                                        sample_point=64
                                        ).cuda()
    feature_s, xyz_s, feature_t1, xyz_t1, feature_t2, xyz_t2, adaptive_xyz = \
        feature_s.cuda(), xyz_s.cuda(), feature_t1.cuda(), xyz_t1.cuda(), feature_t2.cuda(), xyz_t2.cuda(), adaptive_xyz.cuda()
    new_feature_s, new_feature_tf, out_fusion_cls = net_s_trans(feature_s, xyz_s, feature_t1, xyz_t1, feature_t2, xyz_t2, adaptive_xyz)
    print(new_feature_s.shape)