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

if __name__=="__main__":
    #used for test FeatureExtraction
    # net_t = models.__dict__['pointMLP'](num_classes=15)
    # net_s = models.__dict__['pointMLPElite'](num_classes=15)  #
    # net_t = net_t.cuda()
    # net_s = net_s.cuda()
    # train_loader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=4,
    #                           batch_size=2, shuffle=True, drop_last=True)
    # test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=4,
    #                          batch_size=2, shuffle=True, drop_last=False)
    # FeatureExtraction = FeatureExtraction(net_s, net_t)
    # iter_train = iter(train_loader)
    # input = iter_train.__next__()
    # data = input[0]
    # label = input[1]
    # # pt = data[0]
    # # gt = label[0]
    # data = data.permute(0, 2, 1)
    # data = data.cuda()
    # net_t.eval()
    # net_s.eval()
    # net_t.zero_grad()
    # net_s.zero_grad()
    # output_s = net_s(data)
    # output_t = net_t(data)
    # net_tran = ConvBNReLU1D(in_channels=256, out_channels=1024).cuda()
    # output_s_trans = net_tran(FeatureExtraction.s_feature)
    # print(output_s_trans.shape)
    # ------------------------------------- #
    #pos_blocks_list.0.operation.0.net2.0
    #pos_blocks_list.0.operation.1.net2.0
    s_layer_name = "pos_blocks_list.3.operation.0.net2"
    t_layer_name = "pos_blocks_list.3.operation.1.net2"
    net_t = models.__dict__['pointMLP'](num_classes=15)
    net_s = models.__dict__['pointMLPElite'](num_classes=15)  #
    net_t = net_t.cuda()
    net_s = net_s.cuda()
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=4,
                              batch_size=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=4,
                             batch_size=2, shuffle=True, drop_last=False)
    #FeatureExtraction = FeatureExtraction(net_s, net_t)
    CAM_T = CAMExtraction(net_t, mode="T", layer_name=t_layer_name)
    CAM_S = CAMExtraction(net_s, mode="S", layer_name=s_layer_name)
    iter_train = iter(train_loader)
    input = iter_train.__next__()
    data = input[0]
    label = input[1]
    # pt = data[0]
    # gt = label[0]
    data = data.permute(0, 2, 1)
    data = data.cuda()
    data.require_grad = True
    net_t.eval()
    net_s.eval()
    net_t.zero_grad()
    net_s.zero_grad()
    output_s = net_s(data)
    output_t = net_t(data)


    net_tran = ConvBNReLU1D(in_channels=256, out_channels=1024).cuda()
    output_s_trans = net_tran(FeatureExtraction.s_feature)
    print(output_s_trans.shape)