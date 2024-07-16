import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2
'Computes and stores the average and current value.'
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

' network training function '
def train_net(net, device, loader, optimizer, loss_l1,loss_l2, w1,w2,w3,w4,batch_size):
    net.train()
    train_loss = AverageMeter()
    train_loss_fenzi = AverageMeter()   
    train_loss_fenmu = AverageMeter()  
    train_loss_wrap = AverageMeter()  
    train_loss_unwrap = AverageMeter() 
    a1_ave_train_unwrap =AverageMeter() 
    a2_ave_train_unwrap=AverageMeter() 
    a3_ave_train_unwrap=AverageMeter() 
    train_l1loss_unwrap=AverageMeter() 
    train_l2loss_unwrap=AverageMeter()     
    for batch_idx, (input,gt_fenzi,gt_fenmu,gt_wrap,gt_unwrap ) in enumerate(loader):
        input,gt_fenzi,gt_fenmu,gt_wrap,gt_unwrap = input.to(device), gt_fenzi.to(device),gt_fenmu.to(device) ,gt_wrap.to(device),gt_unwrap.to(device)                                                 # Send data to GPU
      #  print("input",input.size(),"gt",gt.size())
        output_train_fenzi,output_train_fenmu,output_train_wrap,output_train_unwrap = net(input) # Forward
        #print("input",input.size(),"gt",gt.size(),"output",output_train.size())

        loss_fenzi = loss_l1(output_train_fenzi, gt_fenzi) +0.5*torch.sqrt(loss_l2(output_train_fenzi, gt_fenzi))
        loss_fenmu=loss_l1(output_train_fenmu, gt_fenmu) +0.5*torch.sqrt(loss_l2(output_train_fenmu, gt_fenmu))
        loss_wrap=loss_l1(output_train_wrap, gt_wrap) +0.5*torch.sqrt(loss_l2(output_train_wrap, gt_wrap))
        #loss_unwrap代表的是l1loss和l2loss的总损失
        loss_unwrap=loss_l1(output_train_unwrap, gt_unwrap)  +0.5*torch.sqrt(loss_l2(output_train_unwrap, gt_unwrap))      
        loss_unwrap_l1=loss_l1(output_train_unwrap, gt_unwrap)
        loss_unwrap_l2=torch.sqrt(loss_l2(output_train_unwrap, gt_unwrap))       
        
        loss=loss_fenzi*w1+loss_fenmu*w2+loss_wrap*w3+loss_unwrap*w4
        
        # 加快收敛速度

        loss.requires_grad_(True)#只有浮点型的损失函数才需要这个
        train_loss.update(loss.item(), output_train_unwrap.size(0)) # Update the record
#其他几个损失函数只更新，显示，并不再单独的梯度下降
        train_loss_fenzi.update(loss_fenzi.item(), output_train_fenzi.size(0)) # Update the record
        train_loss_fenmu.update(loss_fenmu.item(), output_train_fenmu.size(0)) # Update the record
        train_loss_wrap.update(loss_wrap.item(), output_train_wrap.size(0)) # Update the record
        train_loss_unwrap.update(loss_unwrap.item(), output_train_unwrap.size(0)) # Update the record
        train_l1loss_unwrap.update(loss_unwrap_l1.item(), output_train_unwrap.size(0)) # Update the record
        train_l2loss_unwrap.update(loss_unwrap_l2.item(), output_train_unwrap.size(0)) # Update the record
#3sigma准则
        thresh = torch.maximum((gt_unwrap/output_train_unwrap), (output_train_unwrap/gt_unwrap))#thresh[batchsize,1,512,640]

        a1 = (thresh < 1.25).to(torch.float32).mean()#a1 #[batchsize,1,512,640]#已经验证过是对所有通道的a1求平均
        a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
        a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()
        a1_ave_train_unwrap.update(a1.item(),output_train_unwrap.size(0)) #output_val1.size(0)=batchsize
        a2_ave_train_unwrap.update(a2.item(),output_train_unwrap.size(0)) 
        a3_ave_train_unwrap.update(a3.item(),output_train_unwrap.size(0)) 



        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
    output_train_unwrap=output_train_unwrap[-1]
    output_train_unwrap=output_train_unwrap/ (torch.max(output_train_unwrap) - torch.min(output_train_unwrap))
    output_train_fenzi=output_train_fenzi[-1]
    output_train_fenzi=output_train_fenzi/ (torch.max(output_train_fenzi) - torch.min(output_train_fenzi))
    output_train_fenmu=output_train_fenmu[-1]
    output_train_fenmu=output_train_fenmu/ (torch.max(output_train_fenmu) - torch.min(output_train_fenmu))
    output_train_wrap=output_train_wrap[-1]
    output_train_wrap=output_train_wrap/ (torch.max(output_train_wrap) - torch.min(output_train_wrap))
        
    # print(' train_Loss_total: ' + str(round(train_loss.avg, 6)))   
    print(' train_L1loss_unwrap: ' + str(round(train_l1loss_unwrap.avg, 6)))
    print(' train_L2loss_unwrap: ' + str(round(train_l2loss_unwrap.avg, 6)))
    return train_loss_fenzi.avg,output_train_fenzi,\
           train_loss_fenmu.avg,output_train_fenmu,\
           train_loss_wrap.avg,output_train_wrap,\
           train_loss_unwrap.avg,output_train_unwrap,train_loss.avg,\
           a1_ave_train_unwrap.avg,a2_ave_train_unwrap.avg,a3_ave_train_unwrap.avg,\
           train_l1loss_unwrap.avg,train_l2loss_unwrap.avg

' network validating function '
def val_net(net, device, loader,loss_l1,loss_l2,w1,w2,w3,w4,batch_size):

    net.eval()
    val_loss = AverageMeter()
    val_loss_fenzi= AverageMeter() 
    val_loss_fenmu = AverageMeter() 
    val_loss_wrap= AverageMeter() 
    val_loss_unwrap= AverageMeter() 
    a1_ave_val_unwrap = AverageMeter()
    a2_ave_val_unwrap = AverageMeter()
    a3_ave_val_unwrap = AverageMeter()   
    val_l1loss_unwrap=AverageMeter() 
    val_l2loss_unwrap=AverageMeter()    
    with torch.no_grad():
        for batch_idx, (input,gt_fenzi,gt_fenmu,gt_wrap,gt_unwrap ) in enumerate(loader):
            input,gt_fenzi,gt_fenmu,gt_wrap,gt_unwrap = input.to(device), gt_fenzi.to(device),gt_fenmu.to(device),\
            gt_wrap.to(device),gt_unwrap.to(device)                                                 # Send data to GPU
            output_val_fenzi,output_val_fenmu,output_val_wrap,output_val_unwrap = net(input) # Forward
            loss_fenzi_val = loss_l1(output_val_fenzi, gt_fenzi) +0.5*torch.sqrt(loss_l2(output_val_fenzi, gt_fenzi))
            loss_fenmu_val=loss_l1(output_val_fenmu, gt_fenmu) +0.5*torch.sqrt(loss_l2(output_val_fenmu, gt_fenmu))
            loss_wrap_val=loss_l1(output_val_wrap, gt_wrap) +0.5*torch.sqrt(loss_l2(output_val_wrap, gt_wrap))
            loss_unwrap_val=loss_l1(output_val_unwrap, gt_unwrap) +0.5*torch.sqrt(loss_l2(output_val_unwrap, gt_unwrap))
            l1loss_unwrap_val=loss_l1(output_val_unwrap, gt_unwrap)
            l2loss_unwrap_val=torch.sqrt(loss_l2(output_val_unwrap, gt_unwrap) )

            loss=loss_fenzi_val*w1+loss_fenmu_val*w2+loss_wrap_val*w3+loss_unwrap_val*w4
            val_loss.update(loss.item(), output_val_unwrap.size(0)) # Update the record
            val_loss_fenzi.update(loss_fenzi_val.item(), output_val_fenzi.size(0)) # Update the record
            val_loss_fenmu.update(loss_fenmu_val.item(), output_val_fenmu.size(0)) # Update the record
            val_loss_wrap.update(loss_wrap_val.item(), output_val_wrap.size(0)) # Update the record
            val_loss_unwrap.update(loss_unwrap_val.item(), output_val_unwrap.size(0)) # Update the record
            val_l1loss_unwrap.update(l1loss_unwrap_val.item(), output_val_unwrap.size(0)) # Update the record
            val_l2loss_unwrap.update(l2loss_unwrap_val.item(), output_val_unwrap.size(0)) # Update the record
#计算3sigma准则评价
            thresh = torch.maximum((gt_unwrap/output_val_unwrap), (output_val_unwrap/ gt_unwrap))#thresh[batchsize,1,512,640]

            a1 = (thresh < 1.25).to(torch.float32).mean()#a1 #[batchsize,1,512,640]#已经验证过是对所有通道的a1求平均
            a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
            a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()
            a1_ave_val_unwrap.update(a1.item(),output_val_unwrap.size(0)) #output_val1.size(0)=batchsize
            a2_ave_val_unwrap.update(a2.item(),output_val_unwrap.size(0)) 
            a3_ave_val_unwrap.update(a3.item(),output_val_unwrap.size(0)) 


    output_val_fenzi=output_val_fenzi[-1]
    output_val_fenzi=output_val_fenzi/ (torch.max(output_val_fenzi) - torch.min(output_val_fenzi))
    output_val_fenmu=output_val_fenmu[-1]
    output_val_fenmu=output_val_fenmu/ (torch.max(output_val_fenmu) - torch.min(output_val_fenmu))
    output_val_wrap=output_val_wrap[-1]
    output_val_wrap=output_val_wrap/ (torch.max(output_val_wrap) - torch.min(output_val_wrap))
    output_val_unwrap=output_val_unwrap[-1]
    output_val_unwrap=output_val_unwrap/ (torch.max(output_val_unwrap) - torch.min(output_val_unwrap))

    # print(' Val_loss_total: ' + str(round(val_loss.avg, 6)))
    print(' Val_L1loss_unwrap: ' + str(round(val_l1loss_unwrap.avg, 6)))
    print(' Val_L2loss_unwrap: ' + str(round(val_l2loss_unwrap.avg, 6)))
    # print(' a1_ave_val: ' + str(round(a1_ave_val_unwrap.avg, 4)))   
    # print(' a2_ave_val: ' + str(round(a2_ave_val_unwrap.avg, 4)))  
    # print(' a3_ave_val: ' + str(round(a3_ave_val_unwrap.avg, 4)))  
    return val_loss_fenzi.avg, output_val_fenzi,\
    val_loss_fenmu.avg, output_val_fenmu,\
    val_loss_wrap.avg, output_val_wrap,\
    val_loss_unwrap.avg ,output_val_unwrap,val_loss.avg,\
    a1_ave_val_unwrap.avg,a2_ave_val_unwrap.avg,a3_ave_val_unwrap.avg,\
    val_l1loss_unwrap.avg,val_l2loss_unwrap.avg


