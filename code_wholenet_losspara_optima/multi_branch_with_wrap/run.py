import os
     


#再跑公开数据集
for w3 in [1]:#包裹相位的权重
    for w4 in [500,50,10]:
        # if w1 == 1 and w2 == 1 and w3 == 1 and w4 == 10:  # 排除1, 1, 1, 10这组数据

        os.system (f'python /home/lym_pcl/lym_dataset/code_wholenet/code_wholenet_losspara_optima/multi_branch_with_wrap/main_train.py \
                    --w3 {w3} --w4 {w4} \
                    --batch_size 8 --epochs 200  \
                    --wandb_project_name "Nguyen_1523"\
                    --root "/home/lym_pcl/lym_dataset/Nguyen_1523/" \
                    --model "Result/WHOLE_NET/4_10_with_wrap1"')

   



