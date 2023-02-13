"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
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
from kd_losses.st import SoftTarget
from kd_losses.fitnet import Hint, WHint
from kd_losses.RelationD import RelationCos, RKD, LocalRegionMulti, SingleClassifier, TeacherFusion, MultiTeacherOperation
from models.pointmlp import ConvBNReLU1D
from models.hook_feature import FeatureExtraction, FeatureXyzExtractionST, MultiFeatureXyzExtractionST, SingleXyzExtractionST, MultiTeacherFeatureXyzExtractionST
import wandb
def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--s_model', default='pointMLPElite', help='model name [default: pointnet_cls]')
    parser.add_argument('--t_model1', default='pointMLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--t_model2', default='pointMLP', help='model name [default: pointnet_cls]')

    parser.add_argument('--num_classes', default=15, type=int, help='default value for classes of ScanObjectNN')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, default=1834, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')

    # knowledge distillation
    parser.add_argument('--T', type=float, default=4,  help='temperature coefficient')
    parser.add_argument('--t1_model_path', type=str,
                        default="/workspace/model_compress/pointMLP-pytorch/classification_ScanObjectNN/checkpoints/pointMLP-20221026115034/best_checkpoint.pth",
                        help='teacher checkpoint path')
    parser.add_argument('--t2_model_path', type=str,
                        default="/workspace/model_compress/pointMLP-pytorch/classification_ScanObjectNN/checkpoints/pointMLP-20221026115034/best_checkpoint.pth",
                        help='teacher checkpoint path')
    parser.add_argument('--lambda_kd', type=float,default=1,  help='kd hyper-parameter ')
    parser.add_argument('--kd_mode', type=str, default="None", help='kd mode selection ')
    parser.add_argument('--AddST', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--w_dist', type=float, default=5, help='kd hyper-parameter ')
    parser.add_argument('--w_angle', type=float, default=10, help='kd hyper-parameter ')
    parser.add_argument('--k', type=int, default=16, help='kd hyper-parameter ')
    parser.add_argument('--sample_point', type=int, default=32, help='kd hyper-parameter ')
    parser.add_argument('--adaptive_xyz', type=str, default="", help='set the new xyz name')
    parser.add_argument('--name', type=str,default="MultiPointMLP", help='wandb name')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.s_model + 'kd' + message
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')

    net_s = models.__dict__[args.s_model](num_classes=args.num_classes)
    net_t1 = models.__dict__[args.t_model1](num_classes=args.num_classes)
    net_t2 = models.__dict__[args.t_model2](num_classes=args.num_classes)
    criterion = cal_loss
    net_s = net_s.to(device)
    net_t1 = net_t1.to(device)
    net_t2 = net_t2.to(device)
    # criterion = criterion.to(device)
    if device == 'cuda':
        net_s = torch.nn.DataParallel(net_s)
        net_t1 = torch.nn.DataParallel(net_t1)
        net_t2 = torch.nn.DataParallel(net_t2)
        cudnn.benchmark = True
    trainable_list = torch.nn.ModuleList([])
    trainable_list.append(net_s)
    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet_kd" + args.s_model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net_s.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']
    print('==> Loading teacher model..')
    t1_checkpoint_path = args.t1_model_path
    t1_checkpoint = torch.load(t1_checkpoint_path)
    net_t1.load_state_dict(t1_checkpoint['net'])
    print('==> Succeed in teacher model 1..')
    t2_checkpoint_path = args.t2_model_path
    t2_checkpoint = torch.load(t2_checkpoint_path)
    net_t2.load_state_dict(t2_checkpoint['net'])
    print('==> Succeed in teacher model 2..')
    net_s_trans = None
    FExtract = None
    criterionKD = None
    if args.kd_mode == "ST":
        criterionKD = SoftTarget(args.T)
    elif args.kd_mode in [ "FitNet", "CFitNet"]:
        criterionKD = Hint()
        net_s_trans = ConvBNReLU1D(in_channels=512, out_channels=512).to(device)
        net_s_trans = torch.nn.DataParallel(net_s_trans)
        FExtract = FeatureExtraction(net_s, net_t,"module.classifier.0",
                 "module.classifier.0" )
        trainable_list.append(net_s_trans)


    elif args.kd_mode == "WFitNet":
        criterionKD = WHint()
        net_s_trans = ConvBNReLU1D(in_channels=256, out_channels=512).to(device)
        net_s_trans = torch.nn.DataParallel(net_s_trans)
        FExtract = FeatureExtraction(net_s, net_t, "module.pos_blocks_list.2.operation.1.net2",
                                     "module.pos_blocks_list.2.operation.1.net2")
        trainable_list.append(net_s_trans)


    elif args.kd_mode == "RD":
        criterionKD = RKD(args.w_dist, args.w_angle)
        net_s_trans = RelationCos(in_out_channels_s=[256, 1024],
                                  in_out_channels_t=[1024, 1024],
                                  k=args.k,
                                  sample_point=args.sample_point
                                  ).to(device)
        net_s_trans = torch.nn.DataParallel(net_s_trans)
        layer_name_s = "module." + "pos_blocks_list.3.operation.0.net2"
        layer_name_t = "module." + "pos_blocks_list.3.operation.1.net2"
        xyz_s = "module." + "local_grouper_list.3"
        xyz_t = "module." + "local_grouper_list.3"
        if args.adaptive_xyz is "":
            FExtract = FeatureXyzExtractionST(net_s, net_t, layer_name_s=layer_name_s, xyz_name_s=xyz_s,
                                                        layer_name_t=layer_name_t, xyz_name_t=xyz_t)
        else:
            new_xyz = "module." + args.adaptive_xyz
            FExtract = FeatureXyzExtractionST(net_s,
                                              net_t,
                                              layer_name_s=layer_name_s, xyz_name_s=xyz_s,
                                              layer_name_t=layer_name_t, xyz_name_t=xyz_t,
                                              adaptive_xyz_name=new_xyz)
        trainable_list.append(net_s_trans)
    elif args.kd_mode == "RDM":
        criterionKD = RKD(args.w_dist, args.w_angle)
        in_out_channels_s = [[64, 128], [128, 256], [256, 512], [256, 1024]]
        in_out_channels_t = [[128, 128], [256, 256], [512, 512], [1024, 1024]]
        net_s_trans = LocalRegionMulti(in_out_channels_s=in_out_channels_s,
                                       in_out_channels_t=in_out_channels_t).to(device)
        net_s_trans = torch.nn.DataParallel(net_s_trans)
        layer_name_s_list = ["module.pos_blocks_list.0.operation.0.net2", "module.pos_blocks_list.1.operation.0.net2",
                             "module.pos_blocks_list.2.operation.1.net2", "module.pos_blocks_list.3.operation.0.net2"]
        layer_name_t_list = ["module.pos_blocks_list.0.operation.1.net2", "module.pos_blocks_list.1.operation.1.net2",
                             "module.pos_blocks_list.2.operation.1.net2", "module.pos_blocks_list.3.operation.1.net2"]
        xyz_s_list = ["module.local_grouper_list.0", "module.local_grouper_list.1",
                      "module.local_grouper_list.2", "module.local_grouper_list.3"]
        xyz_t_list = ["module.local_grouper_list.0", "module.local_grouper_list.1",
                      "module.local_grouper_list.2", "module.local_grouper_list.3"]
        FExtract = MultiFeatureXyzExtractionST(net_s, net_t,
                                               layer_name_s_list=layer_name_s_list,
                                               layer_name_t_list=layer_name_t_list,
                                               xyz_name_s_list=xyz_s_list,
                                               xyz_name_t_list=xyz_t_list)
        trainable_list.append(net_s_trans)

    elif args.kd_mode == "MT":
        criterionKD = RKD(args.w_dist, args.w_angle)#mse loss
        #约束两个teacher模型的大小，输出两个teacher模型的点结果
        net_s_trans = MultiTeacherOperation(in_channels=[256 ,1024, 1024],
                                  out_channels=[1024, 1024, 1024],
                                  k=args.k,
                                  sample_point=args.sample_point
                                  ).to(device)

        net_s_trans = torch.nn.DataParallel(net_s_trans)
        #约束student和teacher混合后的结果
        layer_name_s = "module." + "pos_blocks_list.3.operation.0.net2"
        layer_name_t1 = "module." + "pos_blocks_list.3.operation.1.net2"
        layer_name_t2 = "module." + "pos_blocks_list.3.operation.1.net2"
        xyz_s = "module." + "local_grouper_list.3"
        xyz_t1 = "module." + "local_grouper_list.3"
        xyz_t2 = "module." + "local_grouper_list.3"
        if args.adaptive_xyz is "":
            FExtract = MultiTeacherFeatureXyzExtractionST(net_s, net_t1, net_t2, layer_name_s=layer_name_s, xyz_name_s=xyz_s,
                                              layer_name_t1=layer_name_t1, xyz_name_t1=xyz_t1,
                                              layer_name_t2=layer_name_t2, xyz_name_t2=xyz_t2)
        else:
            new_xyz = "module." + args.adaptive_xyz
            FExtract = MultiTeacherFeatureXyzExtractionST(net_s, net_t1, net_t2, layer_name_s=layer_name_s, xyz_name_s=xyz_s,
                                              layer_name_t1=layer_name_t1, xyz_name_t1=xyz_t1,
                                              layer_name_t2=layer_name_t2, xyz_name_t2=xyz_t2,
                                              adaptive_xyz_name=new_xyz)
        trainable_list.append(net_s_trans)
        net_t = {'net_t1':net_t1, 'net_t2':net_t2}
    if args.AddST == True:
        criterionKDST = SoftTarget(args.T)
    else:
        criterionKDST = None
    args.name =args.t_model1 + args.t_model2 + args.s_model + str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    wandb.init(name=args.name, project="PointMLP_KD", config=args)
    printf('==> Preparing data..')
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.SGD(trainable_list.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)
    net_ts = {'net_t':net_t, 'net_s':net_s,  'FExtract':FExtract,'net_s_trans':net_s_trans, 'trainable_list':trainable_list}
    criterion_all = {'cls_criterion':criterion, 'kd_criterion':criterionKD, 'criterionKDST':criterionKDST}
    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net_ts, train_loader, optimizer, criterion_all, device, args)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(net_ts, test_loader, criterion_all, device, args)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            net_ts['net_s'], epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # best test accuracy
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


def train(net, trainloader, optimizer, criterion, device, args):
    #net.train()
    net_s = net['net_s']
    net_s.train()
    net_t = None
    net_t1 = None
    net_t2 = None
    if args.kd_mode in ["WFitNet" , "FitNet" , "CFitNet"]:
        net_s_trans = net['net_s_trans']
        FExtract = net['FExtract']
        net_s_trans.train()
    if args.kd_mode not in ["MT"]:
        net_t = net['net_t']
        net_t.eval()
    else:
        net_t1 = net['net_t']['net_t1']
        net_t2 = net['net_t']['net_t2']
        net_t1.eval()
        net_t2.eval()


    cls_criterion = criterion['cls_criterion']
    kd_criterion = criterion['kd_criterion']
    criterionKDST = criterion['criterionKDST']

    if args.kd_mode in ["RD", "RDM", "MT"]:
        net_s_trans = net['net_s_trans']
        FExtract = net['FExtract']
        net_s_trans.train()

    train_loss = 0
    train_kd_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
        optimizer.zero_grad()


        logits_s = net_s(data)
        if args.kd_mode not in ['MT']:
            logits_t = net_t(data)
        else:
            logits_t1 = net_t1(data)
            logits_t2 = net_t2(data)
        # print(logits_t.max(dim=1)[1])
        # print(label)
        # print(sum(logits_t.max(dim=1)[1] ==label).item()/32)
        loss_cls = cls_criterion(logits_s, label)
        loss_kd = 0

        if args.kd_mode == "MT":
            feature_s = FExtract.feature_list_s[data.device].transpose(1, 2)  # [ 2, 256, 64]
            xyz_s = FExtract.xyz_list_s[data.device]  # [ 2, 64, 3 ]

            feature_t1 = FExtract.feature_list_t1[data.device].transpose(1, 2)  # [2, 512, 128]
            feature_t2 = FExtract.feature_list_t2[data.device].transpose(1, 2)  # [2, 512, 128]
            xyz_t1 = FExtract.xyz_list_t1[data.device]  # [2, 128, 3]
            xyz_t2 = FExtract.xyz_list_t2[data.device]
            if args.adaptive_xyz is not "":
                adaptive_xyz = FExtract.adaptive_xyz_list[data.device]
                new_feature_s, new_feature_tf, out_fusion_cls = net_s_trans(feature_s, xyz_s, feature_t1, xyz_t1, feature_t2, xyz_t2, adaptive_xyz)#2,32,256 2,32,3 2,64,1024 | 2, 32, 1024 |2, 128, 3

            else:
                new_feature_s, new_feature_tf, out_fusion_cls = net_s_trans(feature_s, xyz_s, feature_t1, xyz_t1,
                                                                            feature_t2, xyz_t2)

            loss_kd = kd_criterion(new_feature_s, new_feature_tf.detach()) * args.lambda_kd
            loss_kd_cls = cls_criterion(out_fusion_cls, label)
            loss_kd = loss_kd + loss_kd_cls
        if args.kd_mode == "ST":
            loss_kd = kd_criterion(logits_s, logits_t.detach()) * args.lambda_kd

        if args.kd_mode in ["WFitNet" , "FitNet", "CFitNet"]:
            if args.kd_mode == "CFitNet":
                f_s = net_s_trans(FExtract.s_feature_list[data.device].unsqueeze(dim=-1)).squeeze(dim=-1)
                f_t = FExtract.t_feature_list[data.device]
                loss_kd = kd_criterion(f_s, f_t.detach()) * args.lambda_kd
            else:
                f_s = net_s_trans(FExtract.s_feature_list[data.device])
                f_t = FExtract.t_feature_list[data.device]
                loss_kd = kd_criterion(f_s, f_t.detach())*args.lambda_kd

        if args.kd_mode == "RD":
            feature_s = FExtract.feature_list_s[data.device].transpose(1, 2)  # [ 2, 256, 64]
            xyz_s = FExtract.xyz_list_s[data.device]  # [ 2, 64, 3 ]

            feature_t = FExtract.feature_list_t[data.device].transpose(1, 2)  # [2, 512, 128]
            xyz_t = FExtract.xyz_list_t[data.device]  # [2, 128, 3]
            if args.adaptive_xyz is not "":
                adaptive_xyz = FExtract.adaptive_xyz_list[data.device]
                new_feature_s, new_feature_t = net_s_trans(feature_s, xyz_s, feature_t, xyz_t, adaptive_xyz)
            else:
                new_feature_s, new_feature_t = net_s_trans(feature_s, xyz_s, feature_t, xyz_t)
            loss_kd = kd_criterion(new_feature_s, new_feature_t.detach()) * args.lambda_kd

        if args.kd_mode == "RDM":
            feature_s_list, xyz_s_list = FExtract.get_feature_xyz_s(device=data.device)
            feature_t_list, xyz_t_list = FExtract.get_feature_xyz_t(device=data.device)
            new_feature_s_list, new_feature_t_list = net_s_trans(feature_s_list, xyz_s_list, feature_t_list, xyz_t_list)
            loss_kd = (kd_criterion(new_feature_s_list[0], new_feature_t_list[0].detach()) +
                       kd_criterion(new_feature_s_list[1], new_feature_t_list[1].detach()) +
                       kd_criterion(new_feature_s_list[2], new_feature_t_list[2].detach()) +
                       kd_criterion(new_feature_s_list[3], new_feature_t_list[3].detach())) / 4.0 * args.lambda_kd

        loss_kd_ST = 0
        if args.AddST == True:
            loss_kd_ST = criterionKDST(logits_s, logits_t.detach()) * args.lambda_kd

        loss = loss_cls + loss_kd + loss_kd_ST
        loss.backward()
        optimizer.step()
        train_loss += loss_cls.item()
        train_kd_loss += (loss_kd+loss_kd_ST).item()
        preds = logits_s.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()
        wandb.log({
            "train/1.cls_loss": train_loss/ (batch_idx + 1),
            "train/2.kd_loss": train_kd_loss / (batch_idx + 1),
            "train/3.Acc":100. * correct / total
        })
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f|KD: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1),train_kd_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    wandb.log({
        "train/4.epoch_acc": 100. * metrics.accuracy_score(train_true, train_pred),
        "train/5.epoch_acc_avg": 100. * metrics.balanced_accuracy_score(train_true, train_pred),
        "train/6.time":time_cost
    })
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device, args):

    test_loss = 0
    test_kd_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []

    net_s = net['net_s']
    net_s.eval()

    if args.kd_mode in ["WFitNet", "FitNet", "CFitNet"]:
        net_s_trans = net['net_s_trans']
        FExtract = net['FExtract']
        net_s_trans.eval()


    net_t = net['net_t']
    net_t.eval()
    cls_criterion = criterion['cls_criterion']
    kd_criterion = criterion['kd_criterion']
    criterionKDST = criterion['criterionKDST']

    if args.kd_mode in ["RD", "RDM"]:
        net_s_trans = net['net_s_trans']
        FExtract = net['FExtract']
        net_s_trans.eval()
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits_s = net_s(data)
            logits_t = net_t(data)
            cls_loss = cls_criterion(logits_s, label)
            loss_kd = 0
            if args.kd_mode == "ST":
                loss_kd = kd_criterion(logits_s, logits_t.detach()) * args.lambda_kd

            if args.kd_mode in ["WFitNet", "FitNet", "CFitNet"]:
                if args.kd_mode == "CFitNet":
                    f_s = net_s_trans(FExtract.s_feature_list[data.device].unsqueeze(dim=-1)).squeeze(dim=-1)
                    f_t = FExtract.t_feature_list[data.device]
                    loss_kd = kd_criterion(f_s, f_t.detach()) * args.lambda_kd
                else:
                    f_s = net_s_trans(FExtract.s_feature_list[data.device])
                    f_t = FExtract.t_feature_list[data.device]
                    loss_kd = kd_criterion(f_s, f_t.detach()) * args.lambda_kd

            if args.kd_mode == "RD":
                feature_s = FExtract.feature_list_s[data.device].transpose(1, 2)  # [ 2, 256, 64]
                xyz_s = FExtract.xyz_list_s[data.device]  # [ 2, 64, 3 ]

                feature_t = FExtract.feature_list_t[data.device].transpose(1, 2)  # [2, 512, 128]
                xyz_t = FExtract.xyz_list_t[data.device]  # [2, 128, 3]
                # new_feature_s, new_feature_t = net_s_trans(feature_s, xyz_s, feature_t, xyz_t)
                # loss_kd = kd_criterion(new_feature_s, new_feature_t.detach()) * args.lambda_kd
                if args.adaptive_xyz is not "":
                    adaptive_xyz = FExtract.adaptive_xyz_list[data.device]
                    new_feature_s, new_feature_t = net_s_trans(feature_s, xyz_s, feature_t, xyz_t, adaptive_xyz)
                else:
                    new_feature_s, new_feature_t = net_s_trans(feature_s, xyz_s, feature_t, xyz_t)
                loss_kd = kd_criterion(new_feature_s, new_feature_t.detach()) * args.lambda_kd

            if args.kd_mode == "RDM":
                feature_s_list, xyz_s_list = FExtract.get_feature_xyz_s(device=data.device)
                feature_t_list, xyz_t_list = FExtract.get_feature_xyz_t(device=data.device)
                new_feature_s_list, new_feature_t_list = net_s_trans(feature_s_list, xyz_s_list, feature_t_list,
                                                                     xyz_t_list)
                loss_kd = (kd_criterion(new_feature_s_list[0], new_feature_t_list[0].detach()) +
                           kd_criterion(new_feature_s_list[1], new_feature_t_list[1].detach()) +
                           kd_criterion(new_feature_s_list[2], new_feature_t_list[2].detach()) +
                           kd_criterion(new_feature_s_list[3], new_feature_t_list[3].detach())) / 4.0 * args.lambda_kd

            loss_kd_ST = 0

            if args.AddST == True:
                loss_kd_ST = criterionKDST(logits_s, logits_t.detach()) * args.lambda_kd
            #kd_loss = kd_criterion(logits_s, logits_t.detach())
            #loss = cls_loss + loss_kd + loss_kd_ST
            kd_based_loss = loss_kd + loss_kd_ST
            test_loss += cls_loss.item()
            test_kd_loss += kd_based_loss.item()
            preds = logits_s.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            wandb.log({
                "test/1.cls_loss": test_loss / (batch_idx + 1),
                "test/2.kd_loss": test_kd_loss / (batch_idx + 1),
                "test/3.Acc": 100. * correct / total
            })
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f|KD: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1),test_kd_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    wandb.log({
        "test/4.epoch_acc": 100. * metrics.accuracy_score(test_true, test_pred),
        "test/5.epoch_acc_avg": 100. * metrics.balanced_accuracy_score(test_true, test_pred),
        "test/6.time": time_cost
    })
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
