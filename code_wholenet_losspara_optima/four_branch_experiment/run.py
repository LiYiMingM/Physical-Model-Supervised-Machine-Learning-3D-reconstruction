import os
     

   


#再跑公开数据集
for w1 in [1]:#分子的权重设置为0.1和1
    for w2 in [1]:#分母的权重设置为
        for w3 in [1]:#包裹相位的权重
            for w4 in [500,1000]:

                os.system (f'python /home/lym_pcl/lym_dataset/code/multi_branch_experiment/four_branch_erperiment/main_train.py\
                           --w1 {w1} --w2 {w2} --w3 {w3} --w4 {w4} \
                            --batch_size 8 --epochs 200  \
                            --wandb_project_name "Qian_1000"\
                            --root "/home/lym_pcl/lym_dataset/Qian_1000/" \
                            --model "Result/Qian_1000/Wholenet/ResUnet_5_22"')
