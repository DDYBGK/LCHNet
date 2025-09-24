from __future__ import print_function
import os
import sys
import argparse
from urllib.parse import non_hierarchical

import torch
import torch.optim as optim
import tqdm
import provider
import shutil
import importlib
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
#from utils.data_util import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from utils.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
import time
from util.S3DISDataLoader import S3DISDataset
from model.pointMLP import pointMLP

#########################下述废弃不用##################
classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']
#########################上述废弃不用###############

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #C:\Users\13456\Desktop\论文\LCHNet\semantic_seg
ROOT_DIR = BASE_DIR  #C:\Users\13456\Desktop\论文\LCHNet\semantic_seg
sys.path.append(os.path.join(ROOT_DIR, 'model')) #将当前工作目录下的 model 目录加入到 sys.path

#将类别与数字标签对应起来
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}  #enumerate返回列表的索引、再返回classes元素值
seg_classes = class2label
seg_label_to_cat = {}  #创建空字典
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)

'''
    设置 NumPy 的随机种子，以确保后续调用 NumPy 的随机数生成函数时，结果是可复现的。
    随机种子都基于当前时间变化，从而避免不同运行之间的结果完全一致。将两者相加作为随机
    种子，既保证了线程间的独立性，又引入了时间的变化性。
'''
def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))


# '''创建检查点文件夹函数'''
# def _init_():
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists('checkpoints/' + args.exp_name):
#         os.makedirs('checkpoints/' + args.exp_name)

'''
    参数：m是pytorch模型的模块，作用：对模型中的不同类型的层进行权重和偏置的初始化
'''
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(args, io):

    # ============= Model ===================
    #num_part = 50
    num_sem = 13
    device = torch.device("cuda" if args.cuda else "cpu")

    '''
            models.__dict_[]表示取函数或者类的指针,
            这时候后边加上（num_semantic）表示是其参数
            最后.to(device)将这个模型的运行设备进行了指定
            实例化了模型 model就是实例化对象
        '''
    model = pointMLP(num_sem)
    model = model.to(device)
    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    '''
        Resume or not
        这段代码判断是否从之前保存的模型检查点恢复训练
        如果命令行参数 args.resume 为 True，表示需要从之前的训练状态恢复。否则，直接打印 "Training from scratch..."，表示从头开始训练。
   '''
    if args.resume:
        state_dict = torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name,
                                map_location=torch.device('cpu'))['model']
        for k in state_dict.keys():
            if 'module' not in k:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in state_dict:
                    new_state_dict['module.' + k] = state_dict[k]
                state_dict = new_state_dict
            break
        model.load_state_dict(state_dict)

        print("Resume training model...")
        print(torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name).keys())
    else:
        print("Training from scratch...")

    '''加载数据集'''
    # =========== Dataloader =================
    train_data = S3DISDataset(split='train', data_root=args.seg_path, num_point=args.seg_point,
                                 test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("The number of training data is:%d", len(train_data))

    test_data = S3DISDataset(split='test', data_root=args.seg_path, num_point=args.seg_point,
                                test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("The number of test data is:%d", len(test_data))
    #batch_size是24
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4, drop_last=True, pin_memory=True,
                                                  worker_init_fn=worker_init_fn)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=4, drop_last=True)
    # ============= Optimizer ================
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=0)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.scheduler == 'cos':
        print("Use CosLR")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
    else:
        print("Use StepLR")
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    # ============= Training =================
    best_acc = 0.0            #存放最好的Acc
    best_class_iou = 0.0      #存放最好的mIoU
    best_class_acc = 0.0   #存放最好的mAcc_cls
    num_sem = 13

    for epoch in range(args.epochs):
        '''函数已经修改完成，减少了参数num_classes,将num_seg改为使用num_sem'''
        train_epoch(train_loader, model, opt, scheduler, epoch, num_sem, io)
        print("到达test_epoch")
        '''函数修改已完成，减少了参数num_classes,将num_part改为使用num_sem'''
        test_metrics, total_per_cat_iou = test_epoch(test_loader, model, epoch, num_sem, io)

        # 1. when get the best accuracy, save the model:
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc:%.5f' % best_acc)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc}
            torch.save(state, 'checkpoints/%s/best_acc_model.pth' % args.exp_name)
        #2.when get the best class_avg_acc
        if test_metrics['class_avg_acc'] > best_class_acc:
            best_class_acc = test_metrics['class_avg_acc']
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_avg_acc': best_class_acc}
            torch.save(state, 'checkpoints/%s/best_class_avg_acc_model.pth' % args.exp_name)

        # 3. when get the best class_avg_iou, save the model:
        if test_metrics['class_avg_iou'] > best_class_iou:
            best_class_iou = test_metrics['class_avg_iou']
            # print the iou of each class:
            for cat_idx in range(13):
                io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
            io.cprint('Max class avg iou:%.5f' % best_class_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_avg_iou': best_class_iou}
            torch.save(state, 'checkpoints/%s/best_class_avg_iou_model.pth' % args.exp_name)

    # report best acc, ins_iou, cls_iou
    io.cprint('Final Max Acc:%.5f' % best_acc)
    io.cprint('Final Max class avg iou:%.5f' % best_class_iou)
    io.cprint('Final Max class avg acc:%.5f' % best_class_acc)
    # save last model
    state = {
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_class_iou}
    torch.save(state, 'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))


def train_epoch(train_loader, model, opt, scheduler, epoch, num_sem, io):
    train_loss = 0.0
    count = 0.0
    total_correct = 0  # 计算预测正确的标签数量
    total_seen = 0  # 计算总点数
    mIoU = 0.0
    Acc = 0.0
    mAcc_cls = 0.0
    #labelweights = np.zeros(num_sem)  # 初始化为 [0, 0, ..., 0]（形状 (num_sem,)）
    total_seen_class = [0 for _ in range(num_sem)]      # 计算每个类别的点数
    total_correct_class = [0 for _ in range(num_sem)]   # 计算每个类别的预测正确的标签数量
    total_iou_deno_class = [0 for _ in range(num_sem)]  # 计算每个类别的iou分母
    metrics = defaultdict(lambda: 0.0)
    model.train()
    '''
        #points B, N, 9 
        #label  B, N
    '''
    for batch_id, (points, label) in enumerate(train_loader):
        '''数据增强'''
        points = points.data.numpy()  # B, N, C(9)
        _, num_point, _ = points.shape
        print(f"+++++++++++++++++++++point shape+++++++++++++++++{points.shape}")
        points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
        points = torch.Tensor(points)  # B, N, C(9)
        points_float = points.float()
        #label_long = label.long()  #转换为long类型
        #print(f"label---long----shape {label_long.shape}")
        if args.cuda:
            points = points_float.cuda(non_blocking=True) 
            label = label.long().cuda(non_blocking=True)  # 直接在 GPU 上转换类型long.cuda(non_blocking=True)  #将数据加载到CUDA设备上
            print(f"label-------shape {label.shape}")
        points = points.permute(0, 2, 1)
        #print(f"+++++++++++++++++++++point shape+++++++++++++++++{points.shape}")
        seg_pred = model(points)  #B N 13  预测结果
        seg_pred_view = seg_pred.contiguous().view(-1, num_sem)  # B*N,13    展开为B*N,13
        '''
             #view是重塑操作  比如label[B,N],那么view(-1,1)之后就变成了[B*N,1]
             [:, 0]那么这个操作是取张量中每一行的第一个元素，也就是label的第一个元素从而使其变成了[B*N]的向量
        '''
        batch_label = label.view(-1, 1)[:, 0].cpu().data.numpy() #B*N的向量

        pred_val = seg_pred_view.contiguous().cpu().data.numpy()  # B*N,13
        '''计算预测标签与真实值的准确率'''
        pred_val = np.argmax(pred_val, axis=1)  #预测类别的值  B*N
        correct = np.sum((pred_val==batch_label))
        total_correct += correct
        total_seen += (args.batch_size * args.num_points)  # 计算总点数

        '''计算每个类别的标签出现的次数'''
        # tmp获得每个类别出现的次数。假设：batch_label = [0, 1, 2, 0, 1, 1, 2, 0]，num_sem = 2。tmp 得到 [3, 3, 2]
        tmp, _ = np.histogram(batch_label, range(num_sem))
        #labelweights += tmp  #得到每个类的点数

        for l in range(num_sem):
            total_seen_class[l] += np.sum((batch_label == l))
            total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
            total_iou_deno_class[l] += np.sum(((pred_val==l)| (batch_label==l)))
        '''该批次的 所有类别平均IoU'''
        mIoU = np.mean(np.array(total_correct_class)/(np.array(total_iou_deno_class,dtype=float)+1e-6))
        '''该批次的 各个类别平均IoU'''
        final_total_per_cat_iou = np.array(total_correct_class)/(np.array(total_seen_class, dtype=float)+1e-6)
        #print(f"label.view(-1,1)[:, 0].shape{label.view(-1,1)[:, 0].shape}")
        '''计算loss'''
        loss = F.nll_loss(seg_pred_view, label.view(-1,1)[:, 0])
        ''' #表示该批次各个类的精度平均值 是个float值'''
        mAcc_cls = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        '''该批次所有类的accuracy'''
        Acc = np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)

        # Loss backward
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        count += args.batch_size

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5
    io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])

    metrics['accuracy'] = Acc
    metrics['avg_class_IOU'] = mIoU
    metrics['avg_class_Acc'] = mAcc_cls

    outstr = 'Train %d, loss: %f, train acc: %f, train avg_class_iou: %f, train avg_class_acc: %f' % (epoch + 1, train_loss * 1.0 / count,
                                                                           metrics['accuracy'],
                                                                           metrics['avg_class_IOU'],
                                                                           metrics['avg_class_Acc'])
    io.cprint(outstr)
        # # accuracy
        # seg_pred = seg_pred.contiguous().view(-1, num_sem)  # b*n,13
        # label = label_long.view(-1, 1)[:, 0]   # b*n
        # pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        # '''当前批次中预测正确的点'''
        # correct = pred_choice.eq(label.contiguous().data).sum()  # torch.int64: total number of correct-predict pts
        #
        # # sum
        # shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        # count += args.batch_size   # count the total number of samples in each iteration
        # train_loss += loss.item() * args.batch_size
        # accuracy.append(correct.item()/(args.batch_size * num_point))   # append the accuracy of each iteration
        #
        # # Note: We do not need to calculate per_class iou during training


def test_epoch(test_loader, model, epoch, num_sem, io):
    #test_loss = 0.0
    #count = 0.0
    accuracy = []
    mIoU = 0.0
    total_correct = 0  # 计算预测正确的标签数量
    total_seen = 0  # 计算分割点的总数量
    #labelweights = np.zeros(num_sem)
    total_seen_class = [0 for _ in range(num_sem)]
    total_correct_class = [0 for _ in range(num_sem)]
    total_iou_deno_class = [0 for _ in range(num_sem)]
    shape_ious = 0.0
    final_total_per_cat_iou = np.zeros(13).astype(np.float32)
    #final_total_per_cat_seen = np.zeros(13).astype(np.int32)
    #class_tensor = torch.arange(13).view(13, 1)

    metrics = defaultdict(lambda: 0.0)
    model.eval()

    '''更改后的test_epoch函数'''
    for batch_id, (points, label) in enumerate(test_loader):
        batch_size, num_point, _ = points.size()  # B  N  C(9)
        points, label = Variable(points.float()), Variable(label.long())   #label[B,N]
        #points_trans = points.transpose(2, 1)   # B, C(9), N
        points, label = points.cuda(non_blocking=True), label.cuda(non_blocking=True)
        points = points.permute(0, 2, 1)  
        seg_pred = model(points)  # B, N, 13
        pred_val = seg_pred.contiguous().cpu().data.numpy()
        batch_label = label.view(-1, 1)[:, 0].cpu().data.numpy()  #B*N的向量

        '''计算预测标签与真实标签的准确率'''
        pred_val = np.argmax(pred_val, 2)  #预测类别的值
        correct = np.sum((pred_val == batch_label))  #计算该组的预测标签与真实标签相等的数量
        total_correct += correct
        total_seen += (args.batch_size * args.num_points)

        '''计算每个类别的标签出现次数'''
        # tmp获得每个类别出现的次数。假设：batch_label = [0, 1, 2, 0, 1, 1, 2, 0]，num_sem = 2。tmp 得到 [3, 3, 2]
        tmp, _ = np.histogram(batch_label, range(num_sem))
        #labelweights += tmp   #这是该函数的全局变量数组可以用来记录目前为止每个类的点（真实值）出现了多少次

        for l in range(num_sem):  #遍历该样本中每个类别计算类别平均iou
            '''# total_seen_class是一个数组每个元素对应了每个类别的样本数量,真实值'''
            total_seen_class[l] += np.sum((batch_label == l))
            '''# total_correct_class是一个数组每个元素对应了每个类别预测正确的数量，预测值正确数量'''
            total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
            '''# 每个类别IoU的分母（预测结果或真实标签中至少有一个匹配的样本的数量）'''
            total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
            '''各类别的加权平均值'''
            #labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            '''计算所有类相加之后平均的分割IoU（性能）'''
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            '''该批次各个类的IoU  是个列表'''
            final_total_per_cat_iou = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
            '''该批次各个类的精度 是个列表'''
            # Acc_cls = np.array(total_correct_class)/np.array(total_seen_class, dtype=float)
            ''' #表示该批次各个类的精度平均值 是个float值'''
            mAcc_cls = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float)+ 1e-6))
            '''该批次所有类的accuracy'''
            Acc = np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)
            print(f"计算第{batch_id}批次，{l}类")
            # if mIoU>best_iou:
            #     best_iou = mIoU
        seg_pred_to_loss = seg_pred.contiguous().view(-1, num_sem)  #B,N,13->B*N,13
        label_to_loss = label.view(-1, 1)[:, 0]  #B,N->B*N
        # Loss
        loss = F.nll_loss(seg_pred_to_loss.contiguous(), label_to_loss.contiguous())
        #计数器
        count = 0
        count += args.batch_size
    metrics['accuracy'] = Acc
    metrics['class_avg_iou'] = mIoU
    metrics['class_avg_acc'] = mAcc_cls
    #metrics['best_iou'] = best_iou
    outstr = 'Test %d, loss: %f, test acc: %f  test cls_avg_iou: %f  test cls_avg_acc: %f ' % (epoch + 1, loss * 1.0 / count,
                                                                    metrics['accuracy'], metrics['class_avg_iou'],
                                                                    metrics['class_avg_acc'])
    io.cprint(outstr)

    return metrics, final_total_per_cat_iou  #一个字典一个列表


def test(args, io):
    # Dataloader
    test_data = S3DISDataset(split='test', data_root=args.seg_path, num_point=args.seg_point,
                             test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, drop_last=True)

    # Try to load models
    num_part = 50
    num_sem = 13
    device = torch.device("cuda" if args.cuda else "cpu")

    #model = models.__dict__[args.model](num_part).to(device)
    model = pointMLP(num_sem)
    model = model.to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    accuracy = []
    best_iou = 0.0
    mIoU = 0.0
    total_correct = 0  # 计算预测正确的标签数量
    total_seen = 0  # 计算分割点的总数量
    #labelweights = np.zeros(num_sem)
    total_seen_class = [0 for _ in range(num_sem)]
    total_correct_class = [0 for _ in range(num_sem)]
    total_iou_deno_class = [0 for _ in range(num_sem)]
    mAcc_cls = 0.0
    Acc = 0.0
    final_total_per_cat_iou = np.zeros(13).astype(np.float32)

    model.eval()

    '''更改后的test函数'''
    for batch_id, (points, label) in enumerate(test_loader):
        batch_size, num_point, _ = points.size()  # B  N  C(9)
        points, label = Variable(points.float()), Variable(label.long())  # label[B,N]
        points, label = points.cuda(non_blocking=True), label.cuda(non_blocking=True)
        points = points.permute(0, 2, 1)
        with torch.no_grad():
            seg_pred = model(points)  # B, N, 13
        pred_val = seg_pred.contiguous().cpu().data.numpy()
        batch_label = label.view(-1, 1)[:, 0].cpu().data.numpy()  # B*N的向量

        '''计算预测标签与真实标签的准确率'''
        pred_val = np.argmax(pred_val, 2)  # 预测类别的值
        correct = np.sum((pred_val == batch_label))  # 计算该组的预测标签与真实标签相等的数量
        total_correct += correct
        total_seen += (args.batch_size * args.num_points)

        '''计算每个类别的标签出现次数'''
        # tmp获得每个类别出现的次数。假设：batch_label = [0, 1, 2, 0, 1, 1, 2, 0]，num_sem = 2。tmp 得到 [3, 3, 2]
        tmp, _ = np.histogram(batch_label, range(num_sem))
        #labelweights += tmp  # 这是该函数的全局变量数组可以用来记录目前为止每个类的点（真实值）出现了多少次

        for l in range(num_sem):  # 遍历该样本中每个类别计算类别平均iou
            '''# total_seen_class是一个数组每个元素对应了每个类别的样本点的数量,真实值'''
            total_seen_class[l] += np.sum((batch_label == l))
            '''# total_correct_class是一个数组每个元素对应了每个类别预测正确的数量，预测值正确数量'''
            total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
            '''# 每个类别IoU的分母（预测结果或真实标签中至少有一个匹配的样本的数量）'''
            total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
            '''各类别的加权平均值'''
            #labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            '''该批次所有类平均IoU 是个float值'''
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            '''该批次各个类的IoU  是个列表'''
            final_total_per_cat_iou = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
            '''该批次各个类的精度 是个列表'''
            #Acc_cls = np.array(total_correct_class)/np.array(total_seen_class, dtype=float)
            ''' #表示该批次各个类的精度平均值 是个float值'''
            mAcc_cls = np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=float))
            '''该批次所有类的accuracy'''
            Acc = np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)
            '''
                print(f'eval point avg class IoU: %f' % (mIoU))
                print(f'eval point accuracy: %f' % (total_correct / float(total_seen)))
            '''
            if mIoU > best_iou:
                best_iou = mIoU
                '''
                    print(f'eval point avg class IoU: %f' % (mIoU))
                    print(f'eval point accuracy: %f' % (total_correct / float(total_seen)))
                '''
        seg_pred_to_loss = seg_pred.contiguous().view(-1, num_sem)  # B,N,13->B*N,13
        label_to_loss = label.view(-1, 1)[:, 0]  # B,N->B*N
        # Loss
        loss = F.nll_loss(seg_pred_to_loss.contiguous(), label_to_loss.contiguous())
    for l in range(num_sem):
        if total_seen_class[l]>0:
            io.cprint(classes_str[l] + ' iou: ' + str(final_total_per_cat_iou[l]))  # print the iou of each class
    outstr = 'avg class acc: %f  test avg class IOU: %f, avg class acc: %f' % (mAcc_cls, mIoU, Acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Semantic Segmentation')
    parser.add_argument('--model', type=str, default='PointMLP1')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--seg_point', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--seg_path', type=str, default='data/', help='Segment Data Path')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name

    _init_()

    if not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

