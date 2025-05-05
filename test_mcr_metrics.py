import argparse
from ast import arg
import logging
import os
import time
from datasets.MCRDataset import MCRDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # 修改导入语句
import numpy as np
from models.model import *
from datasets.MCRDataset import *
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch
from PIL import Image
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params / 1e6  # 转换为百万（M）

def test(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    # 如果'logs'文件夹不存在，则创建一个
    logs_folder =  os.path.join(args.result_dir, 'logs')
    images_folder =  os.path.join(args.result_dir, 'images')
    
    
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    
    # 设置日志记录
    logging.basicConfig(filename=os.path.join(logs_folder, 'test.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

    dataset = MCRDataset(data_dir='/raid/hbj/datas/MCR',image_list_file='MCR_test_list.txt', data_type='test',patch_size=None)

    print(len(dataset)) 
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    # model
    model = MGCC()
    model.to(device)
    # model = torch.load(args.model)
        # GPU数量大于1才能实现多GPU并行计算
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [32, xxx] -> [8, ...], [8, ...], [8, ...] on 4 GPUs
        # 使用 DataParallel 把模型包装起来
        model = nn.DataParallel(model)
        model.to(device)
    
    # 加载模型权重
    # checkpoint = torch.load(args.model)
    # model.load_state_dict(checkpoint['model'], strict=True)
    
    model.load_state_dict(torch.load(args.model))
    
    model.to(device)
    # total_params = count_parameters(model)
    # print(f'Total trainable parameters: {total_params}')
    model.eval()

    # 获取迭代器的长度
    total_batches = len(test_loader)
    print(total_batches) # 598
    # 初始化 PSNR 和 SSIM 列表
    psnr_list = []
    ssim_list = []
    lpips_list = []
    with torch.no_grad(): 
        # testing
        for i, data in tqdm(enumerate(test_loader), total=total_batches):
            etime = time.time()
            input_path, gt_path, ratio = data['input_path'][0], data['gt_path'][0], data['ratio'][0]
            input_raw = data['input_raw'].cuda(non_blocking=True)
            gt_rgb = data['gt_rgb'].cuda(non_blocking=True)
            preds = model(input_raw)#pred_rgb, pred_raw
            
            pred_rgb = preds[0]
            
            print(pred_rgb.shape)
            pred_rgb = pred_rgb.unsqueeze(0)
            print(pred_rgb.shape)
            print(gt_rgb.shape)
            psnr =  peak_signal_noise_ratio(gt_rgb, preds)
            ssim =  structural_similarity_index_measure(gt_rgb, preds)

            pred_rgb = torch.clamp(pred_rgb, 0, 1)
            gt_rgb = torch.clamp(gt_rgb, 0, 1)
            lpips = get_lpips_torch(pred_rgb, gt_rgb)

            pred_rgb =  pred_rgb * 255
            gt_rgb = gt_rgb * 255
            pred_rgb = pred_rgb.round()
            gt_rgb = gt_rgb.round()


            
    
            # 将 PSNR 和 SSIM 添加到列表中
            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())    
            lpips_list.append(lpips.item())
            # print(f"outputs的大小为{outputs.shape},类型为{type(outputs)}") # outputs的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
            pred_rgb = pred_rgb.squeeze(0).cpu().numpy().astype(np.uint8)

            save_path = os.path.join(images_folder, os.path.basename(input_path)[:-4]) + '_out.jpg'
            Image.fromarray(pred_rgb.transpose(1, 2, 0)).save(save_path)
            
            gt_rgb = gt_rgb.squeeze(0).cpu().numpy().astype(np.uint8)
            save_path = os.path.join(images_folder, os.path.basename(input_path)[:-4]) + '_gt.jpg'
            Image.fromarray(gt_rgb.transpose(1, 2, 0)).save(save_path)
        
            

        # 计算平均 PSNR 和 SSIM
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list) 
        logging.info(f"Average PSNR: {avg_psnr},Average SSIM: {avg_ssim},Average LPIPS: {avg_lpips}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    
    parser = argparse.ArgumentParser(description="evaluating model")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1, help='multi-threads for data loading')
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    test(args)