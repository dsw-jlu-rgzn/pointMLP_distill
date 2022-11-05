import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
from open3d import *
import open3d
import matplotlib.pyplot as plt

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

train_loader = DataLoader(ScanObjectNN(partition='training', num_points=2048), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(ScanObjectNN(partition='test', num_points=2048), num_workers=4,
                             batch_size=32, shuffle=True, drop_last=False)

iter_train = iter(train_loader)
input = iter_train.__next__()
print(input[0].shape)
data = input[0].numpy()
label = input[1].numpy()
out = {'data':data, 'label':label}
np.savez('ScanObjectNN.npz',data, label)
print("hello")

#torch.save(input, 'ScanObjectNN_data.pt')
# source_data = input[0][0]
#
# point_cloud = open3d.geometry.PointCloud()
# point_cloud.points = open3d.utility.Vector3dVector(source_data)
# open3d.visualization.draw_geometries([point_cloud])
# open3d.io.write_point_cloud("copy_of_fragment.pcd", point_cloud)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='y')
# print(points.shape)
# print(points[:, 0].shape, points[:, 1].shape, points[:, 2].shape)
# print(points)
# plt.savefig("matplotlib.png")
# plt.show()
# exit()