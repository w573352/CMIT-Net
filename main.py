import argparse
import os
import time
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from BraTS import get_datasets
from model.CMIT_Net import CMITNet
from DataAugment import DataAugmenter
from utils import mkdir,save_model, save_best_model, save_seg_csv, cal_dice, cal_confuse, save_test_label, AverageMeter, save_checkpoint
from torch.backends import cudnn
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceLoss
from monai.inferers import sliding_window_inference
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

parser = argparse.ArgumentParser(description='BraTS')
parser.add_argument('--exp-name', default="CMIT", type=str)
parser.add_argument('--mode', choices=['train', 'test'], default='test')
parser.add_argument('--dataset-folder',default="Data", type=str, help="Please reference the README file for the detailed dataset structure.")
parser.add_argument('--workers', default=8, type=int, help="The value of CPU's num_worker")
parser.add_argument('--end_epoch', default=3, type=int, help="Maximum iterations of the model")
parser.add_argument('--batch-size', default=1, type=int)
#本来的parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--devices', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--tta', default=True, type=bool, help="test time augmentation")
parser.add_argument('--seed', default=1)
parser.add_argument('--val', default=1, type=int, help="Validation frequency of the model")
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')

def init_randon(seed):
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False         
    cudnn.deterministic = True

def init_folder(args):
    args.base_folder =  mkdir(os.path.dirname(os.path.realpath(__file__)))

    args.dataset_folder = mkdir(os.path.join(args.base_folder, args.dataset_folder))
    args.best_folder = mkdir(f"{args.base_folder}/run/best_model/{args.exp_name}")
    args.writer_folder = mkdir(f"{args.base_folder}/run/writer/{args.exp_name}")
    args.pred_folder = mkdir(f"{args.base_folder}/run/pred/{args.exp_name}")
    args.checkpoint_folder = mkdir(f"{args.base_folder}/run/checkpoint/{args.exp_name}")
    args.csv_folder = mkdir(f"{args.base_folder}/run/csv/{args.exp_name}")
    print(f"The code folder are located in {os.path.dirname(os.path.realpath(__file__))}")
    print(f"The dataset folder located in {args.dataset_folder}")
def main(args):
    model = CMITNet(model_num=4,
                         out_channels=3,
                         image_size=[128,128,128],
                         window_size=(4, 4, 4),
                         ).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)
    criterion=DiceLoss(sigmoid=True).cuda()
    #本来的optimizer=torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=1e-5, amsgrad=True)
    if args.optim_name == 'adam':
        print("采用adam优化器")
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)

    #args.optim_name == 'adamw'
    elif args.optim_name == 'adamw':#采用
        print("采用adamw优化器")
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,#1e-4
                                      weight_decay=args.reg_weight)#1e-5


    if args.mode == "train":
        """
        get_datasets是返回一个字典 t1 t1ce t2 flair
        """
        writer = SummaryWriter(args.writer_folder)
        train_dataset = get_datasets(args.dataset_folder, "train")
        val_dataset = get_datasets(args.dataset_folder, "train_val")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=args.workers)
        print("训练集{} 验证集{}".format(len(train_loader),len(val_loader)))
        train_manager(args, train_loader, val_loader, model, criterion, optimizer, writer)
    
    elif args.mode == "test" :
        print("start test")
        filename = 'model_{}'.format(args.end_epoch)
        file_path = os.path.join(args.best_folder, filename + '.pt')
        print("读取权重文件路径:{}".format(file_path))
        print("文件名为:{}".format(filename))
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        test_dataset = get_datasets(args.dataset_folder, "test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=args.workers)
        test(args, "test", test_loader, model,filename)
   

def train_manager(args, train_loader, val_loader, model, criterion, optimizer, writer):
    best_test_dice = 0
    best_test_epoch = 0
    if args.lrschedule == 'warmup_cosine':#采用
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,#50
                                                  max_epochs=500)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_folder, "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_test_dice = checkpoint['best_test_dice']
        best_test_epoch = checkpoint['best_test_epoch']
    print(f"start train from epoch = {start_epoch}")
    
    for epoch in range(start_epoch, args.end_epoch):#0-500
        model.train()
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        #学习率已经更新
        train_loss = train(args=args,data_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, epoch=epoch, writer=writer)
        if (epoch + 1) % args.val == 0:#args.val = 10
            model.eval()
            with torch.no_grad():
                test_dice = test_val(args=args,data_loader=val_loader, model=model, epoch=epoch, writer=writer)
                if test_dice > best_test_dice:
                    best_test_dice = test_dice
                    best_test_epoch = epoch
                    save_best_model(args, model)
            save_checkpoint(args, dict(epoch=epoch,best_test_epoch=best_test_epoch,best_test_dice=best_test_dice, model = model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()))
        if (epoch + 1) == args.end_epoch:
            filename = 'model_{}.pt'.format(epoch + 1)
            save_model(model=model, epoch=epoch, args=args, filename=filename)
    print("finish train epoch")

def train(args,data_loader, model, criterion, optimizer, scheduler, epoch, writer):
    train_loss_meter = AverageMeter('Loss', ':.4e')
    start_time = time.time()
    for i, data in enumerate(data_loader):
        data_aug = DataAugmenter().cuda()
        label = data["label"].cuda()#[1, 3, 128, 128, 128]
        images = data["image"].cuda()#shape [1, 4, 128, 128, 128]
        images, label = data_aug(images, label)
        #pred.shape[1, 3, 128, 128, 128])
        pred = model(images)
        train_loss = criterion(pred, label)
        train_loss_meter.update(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        print('Epoch {}/{} {}/{}'.format(epoch, args.end_epoch, i, len(data_loader)),
              'loss: {:.4f}'.format(train_loss_meter.avg),
              'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    scheduler.step()
    writer.add_scalar("loss/train", train_loss_meter.avg, epoch)
    return train_loss_meter.avg

def test_val(args,data_loader, model,epoch,writer):
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    dice_sum_list = 0
    length = len(data_loader)
    for i, data in enumerate(data_loader):
        start_time = time.time()
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        targets = data["label"].cuda()
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]
        inputs = inputs.cuda()
        model.cuda()
        with torch.no_grad():
            predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))

        targets = targets[:, :, pad_list[-4]:targets.shape[2] - pad_list[-3],
                  pad_list[-6]:targets.shape[3] - pad_list[-5], pad_list[-8]:targets.shape[4] - pad_list[-7]]
        predict = predict[:, :, pad_list[-4]:predict.shape[2] - pad_list[-3],
                  pad_list[-6]:predict.shape[3] - pad_list[-5], pad_list[-8]:predict.shape[4] - pad_list[-7]]
        predict = (predict > 0.5).squeeze()
        targets = targets.squeeze()
        dice_metrics = cal_dice(predict, targets, haussdor, meandice)
        confuse_metric = cal_confuse(predict, targets, patient_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
        et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
        metrics_dict.append(dict(id=patient_id,
                                 et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice,
                                 et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
                                 et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
                                 et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec))
        dice_sum = (et_dice + tc_dice + wt_dice) / 3
        dice_sum_list = dice_sum_list + dice_sum
        print("正在处理:{} time{:.2f}s".format(patient_id, time.time() - start_time))
    mean_dice = dice_sum_list / length
    writer.add_scalar("dice", mean_dice, epoch)
    return mean_dice

def inference(model, input, batch_size, overlap):
    def _compute(input):
        return sliding_window_inference(inputs=input, roi_size=(128, 128, 128), sw_batch_size=batch_size, predictor=model, overlap=overlap)
    return _compute(input)


def test(args, mode, data_loader, model,filename):
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    for i, data in enumerate(data_loader):
        start_time = time.time()
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        targets = data["label"].cuda()
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]
        inputs = inputs.cuda()
        model.cuda()
        with torch.no_grad():  
            if args.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                predict += torch.sigmoid(
                    inference(model, inputs.flip(dims=(2,)), batch_size=2, overlap=0.6).flip(dims=(2,)))
                predict += torch.sigmoid(
                    inference(model, inputs.flip(dims=(3,)), batch_size=2, overlap=0.6).flip(dims=(3,)))
                predict += torch.sigmoid(
                    inference(model, inputs.flip(dims=(4,)), batch_size=2, overlap=0.6).flip(dims=(4,)))
                predict += torch.sigmoid(
                    inference(model, inputs.flip(dims=(2, 3)), batch_size=2, overlap=0.6).flip(dims=(2, 3)))
                predict += torch.sigmoid(
                    inference(model, inputs.flip(dims=(2, 4)), batch_size=2, overlap=0.6).flip(dims=(2, 4)))
                predict += torch.sigmoid(
                    inference(model, inputs.flip(dims=(3, 4)), batch_size=2, overlap=0.6).flip(dims=(3, 4)))
                predict += torch.sigmoid(
                    inference(model, inputs.flip(dims=(2, 3, 4)), batch_size=2, overlap=0.6).flip(dims=(2, 3, 4)))
                predict = predict / 8.0
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                
        targets = targets[:, :, pad_list[-4]:targets.shape[2]-pad_list[-3], pad_list[-6]:targets.shape[3]-pad_list[-5], pad_list[-8]:targets.shape[4]-pad_list[-7]]
        predict = predict[:, :, pad_list[-4]:predict.shape[2]-pad_list[-3], pad_list[-6]:predict.shape[3]-pad_list[-5], pad_list[-8]:predict.shape[4]-pad_list[-7]]
        predict = (predict>0.5).squeeze()
        targets = targets.squeeze()
        dice_metrics = cal_dice(predict, targets, haussdor, meandice)
        confuse_metric = cal_confuse(predict, targets, patient_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
        et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
        metrics_dict.append(dict(id=patient_id,
            et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice, 
            et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
            et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
            et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec))
        print("正在处理:{} time{:.2f}s".format(patient_id,time.time() - start_time))
    save_seg_csv(args, mode, metrics_dict,filename)
  
def reconstruct_label(image):
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False)*(c3 == True)] = 2
    image[(c1 == True)*(c3 == True)] = 4
    return image

if __name__=='__main__':
    args=parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    if torch.cuda.device_count() == 0:
        raise RuntimeWarning("Can not run without GPUs")
    ##args.seed = 1 固定随机种子
    init_randon(args.seed)
    ##配置arg参数
    init_folder(args)
    torch.cuda.set_device(args.devices)
    main(args)

