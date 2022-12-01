import os
import cv2
import torch
from torch.utils.data import DataLoader
from ScanObjectNN import ScanObjectNN
import numpy as np
import models as models
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from models.hook_feature import CAMExtraction
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
seed = 1
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_embedding(data, label, title):

    fig = plt.figure()
    #ax = plt.subplot(121)
    scatter = plt.scatter(data[:,0], data[:,1], c=label, s=10, alpha=1, cmap=plt.cm.get_cmap('Spectral', 15))
    # for i in range(data.shape[0]):
    #     plt.text(x=data[i, 0], y=data[i, 1],
    #              color=plt.cm.Set1(label[i] / 15.),
    #              fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    plt.title(title)
    return fig

def get_part_data(dataloader, net, FeatExtract):
    feature_list = []
    label_list = []
    for batch_idx, input in enumerate(dataloader):
        data = input[0].cuda()
        label = input[1].cuda()
        data = data.permute(0, 2, 1)
        with torch.no_grad():
            output = net(data)
        feature = FeatExtract.feature_list[data.device]# 8*1024*64
        #x = F.adaptive_max_pool1d(feature, 1).squeeze(dim=-1)
        feature_list.append(feature)
        label_list.append(label)
        if len(label_list) * feature_list[0].shape[2] * feature_list[0].shape[0] > 10000:
            break
        #print("----------------i----------------")
    feature_all = torch.cat(feature_list)
    batch, channel, num_point = feature_all.shape

    new_feature = torch.reshape(feature_all.transpose(1,2), (-1, channel))

    label_all = torch.cat(label_list)
    new_label =label_all.unsqueeze(-1).repeat((1, num_point))
    new_label = torch.reshape(new_label, (-1, ))
    print(new_feature.shape, new_label.shape)
    feature_ = new_feature.cpu().numpy()
    label_ = new_label.cpu().numpy()
    n_samples = feature_.shape[0]
    n_features = feature_.shape[1]
    return feature_, label_, n_samples, n_features


def get_data(dataloader, net, FeatExtract):
    feature_list = []
    label_list = []
    for batch_idx, input in enumerate(dataloader):
        #input = iter_test.__next__()
        data = input[0].cuda()
        label = input[1].cuda()
        data = data.permute(0, 2, 1)
        with torch.no_grad():
            output = net(data)
        feature = FeatExtract.feature_list[data.device]
        x = F.adaptive_max_pool1d(feature, 1).squeeze(dim=-1)
        feature_list.append(x)
        label_list.append(label)
        #print("----------------i----------------")
    feature_all = torch.cat(feature_list)
    label_all = torch.cat(label_list)
    print(feature_all.shape, label_all.shape)
    feature_ = feature_all.cpu().numpy()
    label_ = label_all.cpu().numpy()
    n_samples = feature_.shape[0]
    n_features = feature_.shape[1]
    return feature_, label_, n_samples, n_features


def load_ckpt(net_path, net):
    checkpoint = torch.load(net_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    #net_t = net_t.cuda()
    return net
if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    set_seed(seed)
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=4,
                                  batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=4,
                                 batch_size=16, shuffle=True, drop_last=False)
    net_t = models.__dict__['pointMLP'](num_classes=15)
    net_s = models.__dict__['pointMLPEliteD'](num_classes=15)
    s_checkpoint_path = "/workspace/model_compress/pointMLP-pytorch/classification_ScanObjectNN/checkpoints/pointMLPEliteDkd-20221110164552/best_checkpoint.pth"#
    t_checkpoint_path = "/workspace/model_compress/pointMLP-pytorch/classification_ScanObjectNN/checkpoints/pointMLP-20221026115034/best_checkpoint.pth"
    net_t = load_ckpt(t_checkpoint_path, net_t)
    net_t = net_t.cuda()

    net_s = load_ckpt(s_checkpoint_path, net_s)
    net_s = net_s.cuda()

    FeatExtract = CAMExtraction(net=net_s, layer_name="pos_blocks_list.3.operation.0.net2")#
    feature_, label_, n_samples, n_features = get_part_data(test_loader, net_s, FeatExtract)
    print("feature_ shape:{}".format(feature_.shape))
    print("label_ shape:{}".format(label_.shape))
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(feature_)
    fig = plot_embedding(result, label_, 't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()

