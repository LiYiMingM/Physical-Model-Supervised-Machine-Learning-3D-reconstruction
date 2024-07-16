import os
from optparse import OptionParser
import torch
import time
import os.path
import scipy.io as sio
#from Network_HRNET import HighResolutionNet
#from Network_UNET import UNet
from dataset_read import get_dataloaders
from UNET_RES import UNet ,WHOLE_NET
import csv
' Definition of the needed parameters '
torch.cuda.empty_cache()
from train_func import AverageMeter


def get_args():
    parser = OptionParser()
    #测试时结果保存的位置
    parser.add_option('-e', '--result', dest='result', default="Result/WHOLE_NET/Qian_1000/1_18/test_result/1.0-1.0-10.0-500.0", help='folder of results')
    parser.add_option('-r', '--root', dest='root', default="/home/lym_pcl/lym_dataset/Qian_1000/", help='root directory')
    ####测试时载入的权重
    parser.add_option('-m', '--model', dest='model', default='Result/WHOLE_NET/Qian_1000/1_18/weights_best0---1.0-1.0-10.0-500.0.pth', help='folder for model/weights')

    parser.add_option('-i', '--input1', dest='input1', default='grating_mat', help='folder of input')
    #parser.add_option('-j', '--input2', dest='input2', default='si_hlow_fre_grating', help='folder of input')code1

    parser.add_option('-g', '--ground_truth1', dest='gt_unwrap', default='unwrapped_phase_mat', help='folder of ground truth')
    parser.add_option('-z', '--ground_truth2', dest='gt_fenzi', default='fenzi', help='folder of ground truth')
    parser.add_option('-f', '--ground_truth3', dest='gt_fenmu', default='fenmu', help='folder of ground truth')
    parser.add_option('-w', '--ground_truth4', dest='gt_wrap', default='wrapped_phase', help='folder of ground truth')

    (options, args) = parser.parse_args()
    return options
' Pass inputs through the Res-UNet '
def get_results(load_weights, dir_input1,dir_gt_fenzi, dir_gt_fenmu, dir_gt_wrap, dir_gt_unwrap, resultdir):

    # torch.cuda.set_device(1)#选择GPU0进行单独训练
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")#此处选了第一块GPU，也可以不选
    # net = UNet().to(device)
    use_cuda = torch.cuda.is_available()
#    device = torch.device("GPU" "cpu")#此处选了第一块GPU，也可以不选
    device = torch.device("cuda:1" if use_cuda else "cpu")
    net = WHOLE_NET().to(device)
   # net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # Load old weights
    checkpoint = torch.load(load_weights, map_location='cpu')
    # print(f'weights path: {load_weights}')
    
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key[7:]
        new_state_dict[new_key] = value
  
    # net.load_state_dict(checkpoint['state_dict'])
    net.load_state_dict(new_state_dict)
    # Load the dataset
    _,_,loader = get_dataloaders(dir_input1,dir_gt_fenzi, dir_gt_fenmu, dir_gt_wrap, dir_gt_unwrap, batch_size=10,val_percent=0.1, test_size=0.1)
    # If resultdir does not exists make folder
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    net.eval()
    loss1_ave_fenzi = AverageMeter()
    loss2_ave_fenzi = AverageMeter()
    loss1_ave_fenmu = AverageMeter()
    loss2_ave_fenmu = AverageMeter()
    loss1_ave_wrap = AverageMeter()
    loss2_ave_wrap = AverageMeter()
    loss1_ave_unwrap = AverageMeter()
    loss2_ave_unwrap = AverageMeter()
    loss1_ave_total = AverageMeter()
    loss2_ave_total = AverageMeter()
    a1_ave = AverageMeter()
    a2_ave = AverageMeter()
    a3_ave = AverageMeter()
    with torch.no_grad():
        time1 = 0
        loss_l1 = torch.nn.L1Loss() 
        loss_l2 = torch.nn.MSELoss()#计算测试数据的平均误差和均方差
        path_csv = resultdir + "test_loss and others_all" + ".csv"
        path_csv_unwrap = resultdir + "test_loss_unwrap_only" + ".csv"
        header = ['l1_total_loss','rmse_total_loss', 'loss1_total_ave','loss2_total_ave','time cost now/second',
                  'l1_fenzi_loss','rmse_fenzi_loss', 'loss1_fenzi_ave','loss2_fenzi_ave',
                  'l1_fenmu_loss','rmse_fenmu_loss', 'loss1_fenmu_ave','loss2_fenmu_ave',
                  'l1_wrap_loss','rmse_wrap_loss', 'loss1_wrap_ave','loss2_wrap_ave',    
                  'l1_unwrap_loss','rmse_unwrap_loss', 'loss1_unwrap_ave','loss2_unwrap_ave',
                 
                  ]
        header_unwrap = ['l1_unwrap_loss','rmse_unwrap_loss', 'loss1_unwrap_ave','loss2_unwrap_ave']        
        count=0
        for (input,gt_fenzi,gt_fenmu,gt_wrap,gt_unwrap ) in loader:
             input,gt_fenzi,gt_fenmu,gt_wrap,gt_unwrap = input.to(device), gt_fenzi.to(device),gt_fenmu.to(device) ,gt_wrap.to(device),gt_unwrap.to(device)                                                 # Send data to GPU
             print("input",input.size())
             output_train_fenzi,output_train_fenmu,output_train_wrap,output_train_unwrap = net(input) # Forward
 

             for th in range(0, len(input)):
                time_start = time.time()
                #loss_fenzi
                # print('shape_fenzi',th,output_train_fenzi [th].shape)
                # print('shape_gt',th,gt_fenzi [th].shape)
                loss1_fenzi = loss_l1(output_train_fenzi [th], gt_fenzi[th])
                loss2_fenzi = torch.sqrt(loss_l2(output_train_fenzi [th], gt_fenzi[th]))
                loss1_ave_fenzi.update(loss1_fenzi.item(),1) 
                loss2_ave_fenzi.update(loss2_fenzi.item(),1) 
                #loss_fenmu
                loss1_fenmu= loss_l1(output_train_fenmu[th], gt_fenmu[th])
                loss2_fenmu= torch.sqrt(loss_l2(output_train_fenmu[th], gt_fenmu[th]))
                loss1_ave_fenmu.update(loss1_fenmu.item(),1) 
                loss2_ave_fenmu.update(loss2_fenmu.item(),1) 
                #wrapS
                loss1_wrap= loss_l1(output_train_wrap[th], gt_wrap[th])
                loss2_wrap= torch.sqrt(loss_l2(output_train_wrap[th], gt_wrap[th]))
                loss1_ave_wrap.update(loss1_wrap.item(),1) 
                loss2_ave_wrap.update(loss2_wrap.item(),1) 
                #loss_unwrap
                loss1_unwrap = loss_l1(output_train_unwrap [th], gt_unwrap[th])
                loss2_unwrap = torch.sqrt(loss_l2(output_train_unwrap [th], gt_unwrap[th]))
                loss1_ave_unwrap.update(loss1_unwrap.item(),1) 
                loss2_ave_unwrap.update(loss2_unwrap.item(),1) 
                #loss_total
                loss1_total= loss1_fenzi +loss1_fenmu+loss1_wrap+loss1_unwrap
                loss2_total= loss2_fenzi +loss2_fenmu+loss2_wrap+loss2_unwrap
                loss1_ave_total.update(loss1_total.item(),1) 
                loss2_ave_total.update(loss2_total.item(),1)     
                #3sigma准则
                thresh = torch.maximum((gt_unwrap[th]/output_train_unwrap[th]), (output_train_unwrap[th] / gt_unwrap[th]))
                a1 = (thresh < 1.25).to(torch.float32).mean()
                a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
                a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()
                a1_ave.update(a1.item(),1) 
                a2_ave.update(a2.item(),1) 
                a3_ave.update(a3.item(),1) 
              
                # print(' loss1_ave_total: ' + str(round(loss1_ave_total.avg, 6)))   
                # print(' loss2_ave_total: ' + str(round(loss2_ave_total.avg, 6)))  
                print(' loss1_ave_unwrap: ' + str(round(loss1_ave_unwrap.avg, 6)))   
                print(' loss2_ave_unwrap: ' + str(round(loss2_ave_unwrap.avg, 6)))  
                print(' loss1_ave_wrap: ' + str(round(loss1_ave_wrap.avg, 6)))   
                print(' loss2_ave_wrap: ' + str(round(loss2_ave_wrap.avg, 6))) 
                print(' loss1_ave_fenzi: ' + str(round(loss1_ave_fenzi.avg, 6)))   
                print(' loss2_ave_fenzi: ' + str(round(loss2_ave_fenzi.avg, 6))) 
                print(' loss1_ave_fenmu: ' + str(round(loss1_ave_fenmu.avg, 6)))   
                print(' loss2_ave_fenmu: ' + str(round(loss2_ave_fenmu.avg, 6))) 
                print(' a1_ave: ' + str(round(a1_ave.avg, 4)))   
                print(' a2_ave: ' + str(round(a2_ave.avg, 4)))  
                print(' a3_ave: ' + str(round(a3_ave.avg, 4)))    
                time_cost_now = time.time() - time_start
                values = [loss1_total.item(), loss2_total.item(),round(loss1_ave_total.avg, 6),round(loss2_ave_total.avg, 6),time_cost_now,
                          loss1_fenzi.item(), loss2_fenzi.item(),round(loss1_ave_fenzi.avg, 6),round(loss2_ave_fenzi.avg, 6),
                          loss1_fenmu.item(), loss2_fenmu.item(),round(loss1_ave_fenmu.avg, 6),round(loss2_ave_fenmu.avg, 6),                         
                          loss1_wrap.item(), loss2_wrap.item(),round(loss1_ave_wrap.avg, 6),round(loss2_ave_wrap.avg, 6),                          
                          loss1_unwrap.item(), loss2_unwrap.item(),round(loss1_ave_unwrap.avg, 6),round(loss2_ave_unwrap.avg, 6),    
                          a1.item(), round(a1_ave.avg, 4),  
                          a2.item(), round(a2_ave.avg, 4),   
                          a3.item(), round(a3_ave.avg, 4),                                 
                          ]
                #这个文件里只有unwrap相位的里l1loss和l2loss,以及3sigma指标
                values_unwrap = [                      
                          loss1_unwrap.item(), loss2_unwrap.item(),round(loss1_ave_unwrap.avg, 6),round(loss2_ave_unwrap.avg, 6), 
                          a1.item(), round(a1_ave.avg, 4),  
                          a2.item(), round(a2_ave.avg, 4),   
                          a3.item(), round(a3_ave.avg, 4)                                         
                          ]
#这个网络架构损失函数太多了，不方便找unwrap的
                if os.path.isfile(path_csv) == False:
                    file = open(path_csv, 'w', newline='')
                    writer_csv = csv.writer(file)
                    writer_csv.writerow(header)
                    writer_csv.writerow(values)
                else:
                    file = open(path_csv, 'a', newline='')
                    writer_csv = csv.writer(file)
                    writer_csv.writerow(values)
                file.close()
#为解包裹相位单独写一个文件

                if os.path.isfile(path_csv_unwrap) == False:
                    file = open(path_csv_unwrap, 'w', newline='')
                    writer_csv = csv.writer(file)
                    writer_csv.writerow(header_unwrap)
                    writer_csv.writerow(values_unwrap)
                else:
                    file = open(path_csv_unwrap, 'a', newline='')
                    writer_csv = csv.writer(file)
                    writer_csv.writerow(values_unwrap)
                file.close()




                input1 = input[th]#[1, 256, 256]
                input1_1 = torch.unsqueeze(input1, 0)
                # print('input1_1 ',input1_1.shape)
                output_train_fenzi_1,output_train_fenmu_1,output_train_wrap_1,output_train_unwrap_1= net(input1_1)
                # print("output_train_fenzi shape: ", output_train_fenzi.shape)
                # print("output_train_fenmu shape: ", output_train_fenmu.shape)
                # print("output_train_wrap shape: ", output_train_wrap.shape)
                # print("output_train_unwrap shape: ", output_train_unwrap.shape)
                input1_numpy = input1_1.squeeze(0).squeeze(0).cpu().numpy()
                #fenzi
                output_numpy_fenzi = output_train_fenzi_1.squeeze(0).squeeze(0).cpu().numpy()
                gt_numpy_fenzi = gt_fenzi[th].squeeze(0).cpu().numpy()
                #fenmu
                output_numpy_fenmu = output_train_fenmu_1.squeeze(0).squeeze(0).cpu().numpy()
                gt_numpy_fenmu = gt_fenmu[th].squeeze(0).cpu().numpy()
                #wrap
                output_numpy_wrap = output_train_wrap_1.squeeze(0).squeeze(0).cpu().numpy()
                gt_numpy_wrap = gt_wrap[th].squeeze(0).cpu().numpy()
                #unwrap
                output_numpy_unwrap = output_train_unwrap_1.squeeze(0).squeeze(0).cpu().numpy()
                gt_numpy_unwrap = gt_unwrap[th].squeeze(0).cpu().numpy()


                filename = resultdir + (str(count + 1).zfill(6)) + '-results.mat'
                sio.savemat(filename, {'input': input1_numpy,'output_fenzi': output_numpy_fenzi, 'gt_fenzi': gt_numpy_fenzi,
                                       'output_fenmu': output_numpy_fenmu, 'gt_fenmu': gt_numpy_fenmu ,
                                       'output_wrap': output_numpy_wrap, 'gt_wrap': gt_numpy_wrap,
                                       'output_unwrap': output_numpy_unwrap, 'gt_unwrap': gt_numpy_unwrap,
                                         })
               # print("Result saved as {}".format(filename))
                count +=1
                time_end = time.time()
                time1 = time1 + (time_end - time_start)
        print('totally cost', time1)



' Run the application '
if __name__ == '__main__':
    args = get_args()
    get_results(load_weights = args.root + args.model,

                dir_input1=args.root + args.input1 + '/',
            # dir_input2=args.root + args.input2 + '/',
                dir_gt_unwrap = args.root +args.gt_unwrap+'/',
                dir_gt_fenzi = args.root +args.gt_fenzi+'/',
                dir_gt_fenmu = args.root +args.gt_fenmu+'/',
                dir_gt_wrap = args.root +args.gt_wrap+'/',
                resultdir=args.root + args.result + "/")
    
