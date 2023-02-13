import os
import cv2
import torch
from torch.utils.data import DataLoader
from ScanObjectNN import ScanObjectNN
import numpy as np
import models as models
import datetime
import sklearn.metrics as metrics
from collections import OrderedDict
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
import torch.nn as nn
import argparse
seed = 1
def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--model', default='pointMLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--model_path', default="/home/ftang/swdong/kd_point/pointMLP_distill/classification_ScanObjectNN/checkpoints/pointMLP-20221201210725/best_checkpoint.pth", help='model name [default: pointnet_cls]')
    return parser.parse_args()
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def validate(net, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            # loss = criterion(logits, label)
            # test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
set_seed(seed)
args = parse_args()
train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=4,
                             batch_size=32, shuffle=True, drop_last=False)
#iter_train = iter(train_loader)
#input = iter_train.__next__()
#data = input[0]
#label = input[1]

#data.require_grad = True

net_t = models.__dict__[args.model](num_classes=15)
net_s = models.__dict__['pointMLPElite'](num_classes=15)#
t_checkpoint_path = "/home/ftang/swdong/kd_point/pointMLP_distill/classification_ScanObjectNN/checkpoints/pointMLP-20221201210725/best_checkpoint.pth"
t_checkpoint_path = args.model_path
t_checkpoint = torch.load(t_checkpoint_path)
new_state_dict = OrderedDict()
for k, v in t_checkpoint['net'].items():
    name = k[7:]
    new_state_dict[name] = v
net_t.load_state_dict(new_state_dict)
net_t = net_t.cuda()
device = 'cuda'
#a = get_last_conv_name(net_t)
# data = data.permute(0,2,1)
# data = data.cuda()
# data.permute(0, 2, 1)
layer_name = "pos_blocks_list.0.operation.1.net2.0"
#pos_blocks_list.3.operation.0.net2
#pos_blocks_list.3.operation.1.net2
output =  validate(net_t, test_loader, device)
print("acc:{} acc_avg:{}".format(output["acc"], output["acc_avg"]))
