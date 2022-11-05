import os
import cv2
import torch
from torch.utils.data import DataLoader
from ScanObjectNN import ScanObjectNN
import numpy as np
import models as models
from collections import OrderedDict
import torch.nn as nn

seed = 1
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, xyz_name):
        self.net = net
        self.layer_name = layer_name
        self.xyz_name = xyz_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        #pos_blocks_list.3.operation.1.net2.0
        #local_grouper_list.3
    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        if self.gradient == None:

            self.gradient = output_grad[0]
        else:
            self.gradient = output_grad[0] + self.gradient

    def _get_xyz_features_hook(self, module, input, output):
        self.xyz = output[0]
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
        self.net.zero_grad()
        output = self.net(inputs)  # [2,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        #target = output[0][index[0]]
        #target.backward()
        #target = 0
        for i in range(len(index)):
            #target = target + output[i][index[i]]
            output[i][index[i]].backward(retain_graph=True)
            #target.backward()
        #target.backward()
        #xyz = self.xyz
        # gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        # weight = np.mean(gradient, axis=( 1 ))  # [C]
        #
        # feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
        #
        # cam = feature * weight[:, np.newaxis]  # [C,H,W]
        # cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU
        #
        # # 数值归一化
        # cam -= np.min(cam)
        # cam /= np.max(cam)
        # # resize to 224*224
        # #cam = cv2.resize(cam, (224, 224))

        gradient = self.gradient.cpu().data.numpy()# [B,C,N]
        weight = np.mean(gradient, axis=( 2 )) # [B,C]
        feature = self.feature.cpu().data.numpy() #[B,C,N]
        cam = feature * weight[:, :, np.newaxis] #[B,C,N]
        cam = np.sum(cam, axis=1) #[B,N]
        cam = np.maximum(cam, 0)
        # 数值归一化
        cam -= np.tile(np.min(cam,1)[:,np.newaxis], (1,cam.shape[1]))#.tile(4,512)
        cam /= np.tile(np.max(cam,1)[:,np.newaxis], (1,cam.shape[1]))#.tile(4,512)
        return cam, self.xyz


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        #print(name)
        if isinstance(m, nn.Conv1d):
            print(layer_name)
            layer_name = name
    return layer_name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
set_seed(seed)
train_loader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=4,
                              batch_size=2, shuffle=True, drop_last=True)
test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=4,
                             batch_size=2, shuffle=True, drop_last=False)
iter_train = iter(train_loader)
input = iter_train.__next__()
data = input[0]
label = input[1]
# pt = data[0]
# gt = label[0]
data.require_grad = True
#pt = pt.unsqueeze(0)
#print(pt.shape)
net_t = models.__dict__['pointMLP'](num_classes=15)
net_s = models.__dict__['pointMLPElite'](num_classes=15)#
t_checkpoint_path = "/workspace/model_compress/pointMLP-pytorch/classification_ScanObjectNN/checkpoints/pointMLP-20221026115034/best_checkpoint.pth"
t_checkpoint = torch.load(t_checkpoint_path)
new_state_dict = OrderedDict()
for k, v in t_checkpoint['net'].items():
    name = k[7:]
    new_state_dict[name] = v
net_t.load_state_dict(new_state_dict)
net_t = net_t.cuda()
#a = get_last_conv_name(net_t)
data = data.permute(0,2,1)
data = data.cuda()
data.permute(0, 2, 1)
layer_name = "pos_blocks_list.0.operation.1.net2.0"
#pos_blocks_list.3.operation.0.net2
#pos_blocks_list.3.operation.1.net2
xyz_name = "local_grouper_list.0"
grad_cam = GradCAM(net_t, layer_name=layer_name, xyz_name=xyz_name)
mask, xyz = grad_cam(data, label)
label = label.numpy()
xyz = xyz.cpu().numpy()
print(mask.shape, xyz.shape)
np.savez('ScanObjectNNCAM.npz',xyz, label, mask)

my_dict = {'xzy':xyz, 'label':label, 'mask':mask}
np.save('file.npy', my_dict)
# with torch.no_grad():
#     data = data.permute(0,2,1)
#     data = data.cuda()
#     #data.permute(0, 2, 1)
#     out = net_t(data)