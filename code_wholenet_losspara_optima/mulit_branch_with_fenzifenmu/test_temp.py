# import os
# import numpy as np
# import torch
# import scipy.io as sio
# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader,random_split
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# import random


# input1 = sio.loadmat('/home/lym_pcl/lym_dataset/code_wholenet/code_wholenet_losspara_optima/000002.mat')

# input1 = input1['phi_unwrapped'] #input变量名

# input = torch.from_numpy(input1).float().unsqueeze(0)###将输入图的三个通道（C,H.W）→（1,C,H.W）转换为四个通道，数据是浮点型

# print('111')
import torch

# 文件路径
file_path = '/home/lym_pcl/lym_dataset/Qian_1000/Result/WHOLE_NET/Qian_1000/1_25_bs_8/weights_single---1.0-1.0-1.0-10.0.pth'

# 加载 .pth 文件
data = torch.load(file_path)
print('Epoch:', data['epoch'])
print('Train Loss:', data['train_loss'])
print('Validation Loss:', data['val_loss'])
print('Total Train Loss:', data['train_loss_total'])
print('L1 Loss Validation Unwrap:', data['l1loss_val_unwrap'])
# 打印或检查数据
# print(data)