import argparse
import os
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
import time
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from datasets.MCRDataset import *
import os, time, scipy.io, scipy.misc
import scipy
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from utils.util import *
from models.model import *
from Losses.wavelet_loss import CombinedLoss
from tqdm import tqdm  # 修改导入语句

def train_and_evaluate(args, alpha, beta):
    
    # 创建权重组合子文件夹
    weight_dir = os.path.join(args.result_dir, f'alpha_{alpha}_beta_{beta}')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    save_bestmodel = os.path.join(weight_dir, 'best_loss_model')
    if not os.path.exists(save_bestmodel):
        os.makedirs(save_bestmodel)

    save_lastmodel = os.path.join(weight_dir, 'last_model')
    if not os.path.exists(save_lastmodel):
        os.makedirs(save_lastmodel)
    
    save_best_psnr_model = os.path.join(weight_dir, 'best_psnr_model')
    if not os.path.exists(save_best_psnr_model):
        os.makedirs(save_best_psnr_model)

    save_best_ssim_model = os.path.join(weight_dir, 'best_ssim_model')
    if not os.path.exists(save_best_ssim_model):
        os.makedirs(save_best_ssim_model)

    logs_folder = os.path.join(weight_dir, 'logs') 
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    
# data
    trainset = MCRDataset(args.data_dir, args.train_list_file,data_type='train', patch_size=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,shuffle=True)
    print(f"训练集的总轮数为{len(train_loader)}") # 161

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) # cuda:0

    # 网络模型的加载
    # 加载第一阶段的模型权重
    # 网络模型的加载
    model = MGCC().to(device)
    criterion_wave = CombinedLoss().cuda()
    criterion = torch.nn.L1Loss().cuda()
    

    # GPU数量大于1才能实现多GPU并行计算
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [32, xxx] -> [8, ...], [8, ...], [8, ...] on 4 GPUs
        # 使用 DataParallel 把模型包装起来
        model = nn.DataParallel(model)
        model.to(device)

    if args.resume:   # 如果为True，则表示接着训练；如果为False，则表示从头开始训练
        last_info_list = process_files(save_lastmodel)[0] # 获取列表中的第一个字典
        model_name = last_info_list['文件名']
        model_loss = float(last_info_list['Loss'])
        model_epoch = int(last_info_list['epoch'])  # 将字符串转换为整数
        model_path = os.path.join(save_lastmodel, model_name)    
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)  # 确保模型在GPU上
        lastepoch = model_epoch + 1
        print(f"从{lastepoch}轮继续训练")
        best_info_list = process_files(save_bestmodel)[0]
        min_loss = float(best_info_list['Loss'])

        psnr_info_list = process_files(save_best_psnr_model)[0]
        max_psnr = float(psnr_info_list['Loss'])

        ssim_info_list = process_files(save_best_ssim_model)[0]
        max_ssim = float(ssim_info_list['Loss'])

    else:
        lastepoch = 1
        min_loss = float('inf')
        # val_min_loss = float('inf')  # 初始化最小损失为正无穷
        max_psnr = 0
        max_ssim = 0

    print(f"lastepoch为 {lastepoch}")

    G_opt = optim.Adam(model.parameters(), lr=args.lr)
    # lr scheduler
    scheduler = optim.lr_scheduler.StepLR(G_opt, step_size=100, gamma=0.1)
    
    for epoch in range(lastepoch,args.num_epoch):
        model.train()
        # 如果当前时期的结果目录已经存在，跳过当前时期的训练，继续下一个时期的训练。
        if os.path.isdir(weight_dir + '%04d' % epoch):
            continue
        #Calculating total loss
        etime = time.time()
        eloss = 0
        epsnr = 0
        essim = 0
        
        count = 0
        # 随机排列训练数据的索引，以确保随机性 开始遍历训练数据，对每一张图像进行训练
        # for i, databatch in enumerate(train_loader):
        total_batches = len(train_loader)
        for i, databatch in tqdm(enumerate(train_loader), total=total_batches):
            input_raw = databatch['input_raw'].cuda(non_blocking=True)
  
            gt_rgb = databatch['gt_rgb'].cuda(non_blocking=True)

            
            count += 1

            
            # 将第一阶段的输出传递给第二阶段模型
            preds = model(input_raw)
            
            loss_wave = criterion_wave(preds, gt_rgb)
            loss_L1 = criterion(preds, gt_rgb)
            loss = alpha * loss_L1 + beta * loss_wave
            print(f"Loss损失的值为{loss}")
            
            G_opt.zero_grad()
            loss.backward()
            G_opt.step()
            
            eloss = eloss + loss.item()   # Total Loss
            
            # 计算 PSNR 和 SSIM
            single_psnr =  peak_signal_noise_ratio(gt_rgb, preds)
            single_ssim =  structural_similarity_index_measure(gt_rgb, preds)
            
            epsnr = epsnr + single_psnr.item()  # Total psnr
            essim = essim + single_ssim.item()  # Total SSIM

        # 更新学习率
        scheduler.step()    
       
        print(f"训练一轮总的轮数为{count}")
        aloss = eloss/count
        apsnr = epsnr/count
        assim = essim/count
        temp_loss = aloss
        temp_psnr = apsnr
        temp_ssim = assim
             
                      
        if temp_loss < min_loss:
            min_loss = temp_loss
            min_loss_epoch = epoch
            # 保存最好的权重
            if os.path.exists(save_bestmodel):
                # 先删除再进行保存
                for filename in os.listdir(save_bestmodel):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_bestmodel, filename)
                        os.remove(file_path)
                bestmodel = os.path.join(save_bestmodel, "bestmodel_{}_{}.pth".format(temp_loss, epoch))
                torch.save(model.state_dict(), bestmodel)   
                  
        if temp_psnr > max_psnr:
            max_psnr = temp_psnr
            # 保存最好的权重
            if os.path.exists(save_best_psnr_model):
                # 先删除再进行保存
                for filename in os.listdir(save_best_psnr_model):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_best_psnr_model, filename)
                        os.remove(file_path)
                bestpsnrmodel = os.path.join(save_best_psnr_model, "bestpsnrmodel_{}_{}.pth".format(temp_psnr, epoch))
                torch.save(model.state_dict(), bestpsnrmodel) 
        
        if temp_ssim > max_ssim:
            max_ssim = temp_ssim
            # 保存最好的权重
            if os.path.exists(save_best_ssim_model):
                # 先删除再进行保存
                for filename in os.listdir(save_best_ssim_model):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_best_ssim_model, filename)
                        os.remove(file_path)
                bestssimmodel = os.path.join(save_best_ssim_model, "bestssimmodel_{}_{}.pth".format(temp_ssim, epoch))
                torch.save(model.state_dict(), bestssimmodel) 
        
        print(f"\nEpoch = {epoch}. \tLoss = {aloss}, \tPSNR = {apsnr}, \tSSIM = {assim},\tTime = {time.time() - etime}")
        
        # 保存最后一轮的权重
        # 检查目录中是否有.pth结尾的文件
        if os.path.exists(save_lastmodel):
            # 先删除再进行保存
            for filename in os.listdir(save_lastmodel):
                if filename.endswith('.pth'):
                    # 删除该文件
                    file_path = os.path.join(save_lastmodel, filename)
                    os.remove(file_path)
            lastmodel = os.path.join(save_lastmodel, "ModelSnapshot_{}_{}.pth".format(temp_loss, epoch))
            torch.save(model.state_dict(), lastmodel)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='/data/Zhaobo/MCR')
    parser.add_argument('--train_list_file', type=str, default='MCR_train_list.txt')
    parser.add_argument('--test_list_file',  type=str, default='MCR_test_list.txt')
    
    parser.add_argument('--train_datatype', type=str, default='train')
    parser.add_argument('--test_datatype',  type=str, default='test')
    parser.add_argument('--result_dir', type=str, default='/data/Zhaobo/result/MCR')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=4001)
    parser.add_argument('--model_save_freq', type=int, default=200)
    parser.add_argument('--resume', type=bool, default=True, help='continue training')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    
    train_and_evaluate(args, 0.5, 0.5)