import os
import cv2
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
import numpy as np
import os
import cv2
import torch
from torch.utils.data import DataLoader
from ScanObjectNN import ScanObjectNN
import numpy as np
import models as models
from models.pointmlp import ConvBNReLU1D
from collections import OrderedDict
import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
#from pointmlp import index_points, farthest_point_sample, knn_point
from kd_losses.RelationD import RelationCos, RKD, LocalRegionMulti
from models.pointmlp import index_points, farthest_point_sample, knn_point


'''
target: calculate the Grad-CAM between T and S.
The simple version: directly calculate the Grad similarity.
输入T和S模型，以及point cloud和他的label，输出T,S模型的Grad
Grad如何获得? 通过计算feature map的gradient
'''
class GradCAMST(object):
    def __init__(self, net_s, net_t, layer_name_s, layer_name_t, xyz_name_s, xyz_name_t):
        self.GradCAM_s = GradCAM(net_s, layer_name_s, xyz_name_s)
        self.GradCAM_t = GradCAM(net_t, layer_name_t, xyz_name_t)
    def __call__(self, inputs, index):
        gradient_s, xyz_s = self.GradCAM_s(inputs, index)
        gradient_t, xyz_t = self.GradCAM_t(inputs, index)
        return gradient_s, xyz_s, gradient_t, xyz_t


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, xyz_name):
        self.net = net
        self.layer_name = layer_name
        self.xyz_name = xyz_name
        # self.feature = None
        # self.gradient = None
        self.feature = {}
        self.gradient = {}
        self.xyz = {}
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature[output.device] = output


    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient[output_grad.device] = output_grad

    def _get_xyz_features_hook(self, module, input, output):
        self.xyz[output[0].device] = output[0]
        print(output[0].shape)


    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
            if name == self.xyz_name:
                self.handlers.append(module.register_forward_hook(self._get_xyz_features_hook))
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        #self.net.zero_grad()
        output = self.net(inputs)  # [2,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = 0
        for i in range(len(index)):
            target = target + output[i][index[i]]
            #output[i][index[i]].backward(retain_graph=True)
        target.backward(retain_graph=True)


        #gradient = self.gradient.cpu().data.numpy()# [B,C,N]
        gradient = self.gradient[output.device]

        # weight = np.mean(gradient, axis=( 2 )) # [B,C]
        # feature = self.feature.cpu().data.numpy() #[B,C,N]
        # cam = feature * weight[:, :, np.newaxis] #[B,C,N]
        # cam = np.sum(cam, axis=1) #[B,N]
        # cam = np.maximum(cam, 0)
        # # 数值归一化
        # cam -= np.tile(np.min(cam,1)[:,np.newaxis], (1,cam.shape[1]))#.tile(4,512)
        # cam /= np.tile(np.max(cam,1)[:,np.newaxis], (1,cam.shape[1]))#.tile(4,512)
        #return cam, self.xyz
        return gradient, self.xyz


class MultiFeatureXyzExtractionST(object):
    def __init__(self,
                 net_s,
                 net_t,
                 layer_name_s_list=None,
                 xyz_name_s_list=None,
                 layer_name_t_list=None,
                 xyz_name_t_list=None):
        self.net_s = net_s
        self.net_t = net_t
        self.FeatureXzyExtractionList = []
        for i in range(len(layer_name_t_list)):
            FeatureXyzExt = FeatureXyzExtractionST(self.net_s, self.net_t,
                                                   layer_name_s=layer_name_s_list[i],
                                                   xyz_name_s=xyz_name_s_list[i],
                                                   layer_name_t=layer_name_t_list[i],
                                                   xyz_name_t=xyz_name_t_list[i])
            self.FeatureXzyExtractionList.append(FeatureXyzExt)

    def get_feature_xyz_s(self, device):
        feature_s_list = []
        xyz_s_list = []
        for i in range(len(self.FeatureXzyExtractionList)):
            feature_s = self.FeatureXzyExtractionList[i].feature_list_s[device]
            xyz_s = self.FeatureXzyExtractionList[i].xyz_list_s[device]
            feature_s_list.append(feature_s.transpose(1, 2))
            xyz_s_list.append(xyz_s)
        return feature_s_list, xyz_s_list

    def get_feature_xyz_t(self, device):
        feature_t_list = []
        xyz_t_list = []
        for i in range(len(self.FeatureXzyExtractionList)):
            feature_t = self.FeatureXzyExtractionList[i].feature_list_t[device]
            xyz_t = self.FeatureXzyExtractionList[i].xyz_list_t[device]
            feature_t_list.append(feature_t.transpose(1, 2))
            xyz_t_list.append(xyz_t)
        return feature_t_list, xyz_t_list


class FeatureXyzExtractionST(object):
    def __init__(self, net_s, net_t, layer_name_s="pos_blocks_list.3.operation.0.net2", xyz_name_s="local_grouper_list.3",
                 layer_name_t="pos_blocks_list.3.operation.0.net2", xyz_name_t="local_grouper_list.3"):
        self.net_s = net_s
        self.layer_name_s = layer_name_s
        self.xyz_name_s = xyz_name_s
        self.feature_list_s = {}
        self.xyz_list_s = {}
        self.net_t = net_t
        self.layer_name_t = layer_name_t
        self.xyz_name_t = xyz_name_t
        self.feature_list_t = {}
        self.xyz_list_t = {}
        self.handlers = []
        self._register_hook()

    def _get_features_s_hook(self, module, input, output):
        self.feature_list_s[input[0].device] = output

    def _get_xyz_features_s_hook(self, module, input, output):
        self.xyz_list_s[input[0].device] = output[0]

    def _get_features_t_hook(self, module, input, output):
        self.feature_list_t[input[0].device] = output

    def _get_xyz_features_t_hook(self, module, input, output):
        self.xyz_list_t[input[0].device] = output[0]

    def _register_hook(self):
        for (name, module) in self.net_s.named_modules():
            if name == self.layer_name_s:
                self.handlers.append(module.register_forward_hook(self._get_features_s_hook))
            if name == self.xyz_name_s:
                self.handlers.append(module.register_forward_hook(self._get_xyz_features_s_hook))
        for (name, module) in self.net_t.named_modules():
            if name == self.layer_name_t:
                self.handlers.append(module.register_forward_hook(self._get_features_t_hook))
            if name == self.xyz_name_t:
                self.handlers.append(module.register_forward_hook(self._get_xyz_features_t_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()


class FeatureXyzExtraction(object):
    def __init__(self, net, layer_name="pos_blocks_list.3.operation.0.net2", xyz_name="local_grouper_list.3"):
        self.net = net
        self.layer_name = layer_name
        self.xyz_name = xyz_name
        self.feature_list = {}
        self.xyz_list = {}
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        # if input[0].device not in self.s_feature_list.keys():
        #     self.s_feature_list[input[0].device] = []
        self.feature_list[input[0].device] = output
        #print("s_feature shape:{}".format(output.size()))

    def _get_xyz_features_hook(self, module, input, output):
        self.xyz_list[input[0].device] = output[0]
        #print(output[0].shape)


    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
            if name == self.xyz_name:
                self.handlers.append(module.register_forward_hook(self._get_xyz_features_hook))


    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()


class FeatureExtraction(object):
    def __init__(self, s_net, t_net, s_layer_name="pos_blocks_list.3.operation.0.net2",
                 t_layer_name="pos_blocks_list.3.operation.1.net2"):
        self.s_net = s_net
        self.t_net = t_net
        self.s_feature_list = {}
        self.t_feature_list = {}
        self.s_layer_name = s_layer_name
        self.t_layer_name = t_layer_name
        self.handlers = []
        self._register_hook()

    def _get_s_features_hook(self, module, input, output):
        # if input[0].device not in self.s_feature_list.keys():
        #     self.s_feature_list[input[0].device] = []
        self.s_feature_list[input[0].device] = output
        #print("s_feature shape:{}".format(output.size()))
    def _get_t_features_hook(self, module, input, output):
        self.t_feature_list[input[0].device] = output
        #print("t_feature shape:{}".format(output.size()))
    def _register_hook(self):
        for (name, module) in self.s_net.named_modules():
            if name == self.s_layer_name:
                print("student")
                self.handlers.append(module.register_forward_hook(self._get_s_features_hook))
        for (name, module) in self.t_net.named_modules():
            if name == self.t_layer_name:
                print("teacher")
                self.handlers.append(module.register_forward_hook(self._get_t_features_hook))
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()


class CAMExtraction(object):
    def __init__(self, net, mode="T",  layer_name="pos_blocks_list.3.operation.0.net2",
                 ):
        self.net = net
        self.feature_list = {}
        self.gradient_list = {}
        self.layer_name = layer_name
        self.handlers = []
        self._register_hook()
        self.mode = mode
    def _get_features_hook(self, module, input, output):
        # if input[0].device not in self.s_feature_list.keys():
        #     self.s_feature_list[input[0].device] = []
        self.feature_list[input[0].device] = output
        #print("s_feature shape:{}".format(output.size()))


    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        # if self.gradient == None:
        #     self.gradient = output_grad[0]
        self.gradient_list[input_grad[0].device] = output_grad[0]



    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                print("student")
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
    def __call__(self,output, index):
        target = 0
        for i in range(len(index)):
            target = target + output[i][index[i]]
        target.backward()
        gradient = self.gradient_list[output.device]
        weight, _ = torch.mean(gradient, 2)
        feature = self.feature_list[output.device]
        cam = feature * weight[:, :, None]
        cam, _ = torch.sum(cam, 1)
        zero = torch.zeros_like(cam)
        cam, _ = torch.max(cam, zero)
        return cam


# def cosine_similarity(x1, eps=1e-8):
#     w1 = x1.norm(p=2, dim=-1, keepdim=True)
#     w2 = x1.norm(p=2, dim=-1, keepdim=True)
#     return 1 - torch.bmm(w1, w2.transpose(1,2)) / (w1 * w2.transpose(1,2)).clamp(min=eps)


def cal_cosine_similarity(x):
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise
if __name__=="__main__":
    s_layer_name = "pos_blocks_list.3.operation.0.net2"
    t_layer_name = "pos_blocks_list.3.operation.1.net2"
    s_xyz = "local_grouper_list.3"
    t_xyz = "local_grouper_list.3"
    net_t = models.__dict__['pointMLP'](num_classes=15)
    net_s = models.__dict__['pointMLPEliteD'](num_classes=15)  #
    net_t = net_t.cuda()
    net_s = net_s.cuda()
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=4,
                              batch_size=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=4,
                             batch_size=2, shuffle=True, drop_last=False)
    #FeatureExtraction = FeatureExtraction(net_s, net_t)
    # FeatureXyzExtraction_s = FeatureXyzExtraction(net_s, layer_name=s_layer_name, xyz_name=s_xyz)
    # FeatureXyzExtraction_t = FeatureXyzExtraction(net_t, layer_name=t_layer_name, xyz_name=t_xyz)

    iter_train = iter(train_loader)
    input = iter_train.__next__()
    data = input[0]
    label = input[1]
    # pt = data[0]
    # gt = label[0]
    data = data.permute(0, 2, 1)
    data = data.cuda()
    criterionKD = RKD()
    in_out_channels_s = [[64, 128], [128, 256], [256, 512], [256, 1024]]
    in_out_channels_t = [[128, 128], [256, 256], [512, 512], [1024, 1024]]
    net_s_trans = LocalRegionMulti(in_out_channels_s=in_out_channels_s,
                                   in_out_channels_t=in_out_channels_t)
    net_s_trans = net_s_trans.cuda()
    #net_s_trans = torch.nn.DataParallel(net_s_trans)
    layer_name_s_list = ["pos_blocks_list.0.operation.0.net2", "pos_blocks_list.1.operation.0.net2",
                         "pos_blocks_list.2.operation.1.net2", "pos_blocks_list.3.operation.0.net2"]
    layer_name_t_list = ["pos_blocks_list.0.operation.1.net2", "pos_blocks_list.1.operation.1.net2",
                         "pos_blocks_list.2.operation.1.net2", "pos_blocks_list.3.operation.1.net2"]
    xyz_s_list = ["local_grouper_list.0", "local_grouper_list.1",
                  "local_grouper_list.2", "local_grouper_list.3"]
    xyz_t_list = ["local_grouper_list.0", "local_grouper_list.1",
                  "local_grouper_list.2", "local_grouper_list.3"]
    FExtract = MultiFeatureXyzExtractionST(net_s, net_t,
                                           layer_name_s_list=layer_name_s_list,
                                           layer_name_t_list=layer_name_t_list,
                                           xyz_name_s_list=xyz_s_list,
                                           xyz_name_t_list=xyz_t_list)


    output_s = net_s(data)
    output_t = net_t(data)
    feature_s_list, xyz_s_list = FExtract.get_feature_xyz_s(device=data.device)
    feature_t_list, xyz_t_list = FExtract.get_feature_xyz_t(device=data.device)
    out_point_s_list, out_point_t_list = net_s_trans(feature_s_list, xyz_s_list, feature_t_list, xyz_t_list)

    print(out_point_s_list)
    print("hello")
    # print("hello")
    #
    # RelationCos = RelationCos().cuda()
    # RKD = RKD(15, 20)
    # feature_s = FeatureXyzExtraction_s.feature_list[data.device].transpose(1,2) # [ 2, 256, 64]
    # xyz_s = FeatureXyzExtraction_s.xyz_list[data.device] # [ 2, 64, 3 ]
    #
    # feature_t = FeatureXyzExtraction_t.feature_list[data.device].transpose(1,2) # [2, 512, 128]
    # xyz_t = FeatureXyzExtraction_t.xyz_list[data.device] # [2, 128, 3]
    # feat_s, feat_t = RelationCos(feature_s, xyz_s, feature_t, xyz_t)
    # loss = RKD(feat_s, feat_t)
    # fps_idx_t = pointnet2_utils.furthest_point_sample(xyz_t, 64).long()
    # new_xyz_t = index_points(xyz_t, fps_idx_t)
    # new_points_t = index_points(feature_t, fps_idx_t)
    # idx_t = knn_point(16, xyz_t, new_xyz_t)
    #
    # grouped_xyz_t = index_points(xyz_t, idx_t)  # [B, npoint, k, 3]
    # grouped_points_t = index_points(feature_t, idx_t)  # [2, 64, 16, 128]
    #
    # idx_s = knn_point(16, xyz_s, new_xyz_t)
    # grouped_xyz_s = index_points(xyz_s, idx_s) # [2, 64, 16, 3]
    # grouped_points_s = index_points(feature_s, idx_s) # [2, 64, 16, 512]
    # from pointmlp import ConvBNReLU1D
    # GraphConv_s = ConvBNReLU1D(in_channels=256, out_channels=512).cuda()
    # GraphCOnv_t = ConvBNReLU1D(in_channels=512, out_channels=512).cuda()
    # x = grouped_points_s
    # b, n, s, d = x.size()
    # x = x.permute(0, 1, 3, 2)
    # x = x.reshape(-1, d, s)
    # x = GraphConv_s(x)
    # batch_size, _, _ = x.size()
    # x = torch.nn.functional.adaptive_max_pool1d(x, 1).view(batch_size, -1)
    # x = x.reshape(b, n, -1)
    # print("hello")


