import os
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random



' Define the reading of the dataset '
class PUDataset(Dataset):
    def __init__(self, ids, dir_input1,dir_gt_fenzi, dir_gt_fenmu, dir_gt_wrap, dir_gt_unwrap,extension='.mat'):
        self.dir_input1= dir_input1
        self.dir_gt_fenzi = dir_gt_fenzi
        self.dir_gt_fenmu = dir_gt_fenmu
        self.dir_gt_wrap = dir_gt_wrap
        self.dir_gt_unwrap = dir_gt_unwrap
        self.extension = extension
        self.ids = ids # Dataset IDS
        self.data_len = len(self.ids) # Calculate len of data

    ' Ask for input and ground truth'
    def __getitem__(self, index):

        # Get an ID of the input and ground truth
        id_input1 = self.dir_input1 + self.ids[index] + self.extension
        id_gt_fenzi = self.dir_gt_fenzi + self.ids[index] + self.extension
        id_gt_fenmu = self.dir_gt_fenmu + self.ids[index] + self.extension
        id_gt_wrap = self.dir_gt_wrap + self.ids[index] + self.extension
        id_gt_unwrap = self.dir_gt_unwrap + self.ids[index] + self.extension
        # Open them
        #print(id_input)
        input1 = sio.loadmat(id_input1)
        gt_fenzi = sio.loadmat(id_gt_fenzi)
        gt_fenmu = sio.loadmat(id_gt_fenmu)
        gt_wrap = sio.loadmat(id_gt_wrap)
        gt_unwrap = sio.loadmat(id_gt_unwrap)
        input1 = input1['high_grating'] #input变量名
       # input2 = input2['si'] #input变量名_低频光栅
        gt_fenzi = gt_fenzi['fenzi']   #gt
        gt_fenmu = gt_fenmu['fenmu']   #gt
        gt_wrap = gt_wrap['phi_wrapped']   #gt
        gt_unwrap = gt_unwrap['phi_unwrapped']   #gt

        input = torch.from_numpy(input1).float().unsqueeze(0)###将输入图的三个通道（C,H.W）→（1,C,H.W）转换为四个通道，数据是浮点型
        gt_fenzi = torch.from_numpy(gt_fenzi).float().unsqueeze(0)
        gt_fenmu = torch.from_numpy(gt_fenmu).float().unsqueeze(0)
        gt_wrap = torch.from_numpy(gt_wrap).float().unsqueeze(0)
        gt_unwrap = torch.from_numpy(gt_unwrap).float().unsqueeze(0)



        return input,gt_fenzi,gt_fenmu,gt_wrap,gt_unwrap 

    ' Length of the dataset '
    def __len__(self):
        return self.data_len
    




' Return the training dataset separated in batches '
def get_dataloaders(dir_input1,dir_gt_fenzi, dir_gt_fenmu, dir_gt_wrap, dir_gt_unwrap, batch_size,val_percent=0.1, test_size=0.1):##将数据集的20%分成验证集
    val_percent = val_percent / 100 if val_percent > 1 else val_percent  # Validate a correct percentage
    ids = [f[:-4] for f in os.listdir(dir_input1)] # Read the names of the images
    ids = sorted(ids, key=lambda x: int(x))#对文件夹中的文件按数字进行排序，主要我要分析不同形状之间的误差大小

    dset = PUDataset(ids,dir_input1,dir_gt_fenzi, dir_gt_fenmu, dir_gt_wrap, dir_gt_unwrap) # Get the dataset
    #这一步是让训练集，测试集，验证集每次分配的内容固定
    torch.manual_seed(0)   

    train_dataset, val_dataset ,test_dataset,_= random_split(dset, [0.4, 0.05,0.05,0.5])
   # Create the dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last = False)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,drop_last = False)
   
    # print('len(num_testdata):',len(test_dataset))
    ba=int((len(ids)-1)/100)
    print('ba:',ba)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=ba)

    return dataloaders['train'], dataloaders['val'],dataloaders['test'] 

' Return the dataset for testing '
# def get_dataloader_for_test(dir_input1,dir_gt1,test_size=0.1):
#     ids = [f[:-4] for f in os.listdir(dir_input1)] # Read the names of the datas
#     ids = sorted(ids, key=lambda x: int(x))#对文件夹中的文件按数字进行排序，主要我要分析不同形状之间的误差大小
#     dset = PUDataset(ids,dir_input1,dir_gt1) # Get the dataset
#     train_dataset,test_dataset=train_test_split(dset,test_size=test_size,random_state=42)  


#     print('len(ids):',len(test_dataset))
#     ba=int((len(ids)+3)/4)
#     print('ba:',ba)
#     dataloader = DataLoader(test_dataset, batch_size=ba) # Create the dataloader
#     return dataloader