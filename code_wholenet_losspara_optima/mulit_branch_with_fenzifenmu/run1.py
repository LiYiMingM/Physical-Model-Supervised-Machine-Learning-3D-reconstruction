import os


# for w1 in [1]:#分子的权重设置为0.1和1
#     for w2 in [1]:#分母的权重设置为
#             for w4 in [500,2000]:

#                 os.system (f'python /home/lym_pcl/lym_dataset/code_wholenet/code_wholenet_losspara_optima/mulit_branch_with_fenzifenmu/main_train.py\
#                            --w1 {w1} --w2 {w2} --w4 {w4} \
#                             --batch_size 8 --epochs 200  \
#                             --wandb_project_name "Nguyen_1523"\
#                             --root "/home/lym_pcl/lym_dataset/Nguyen_1523/" \
#                             --model "Result/WHOLE_NET/4_9_withfenzi_fenmu1"')


for w1 in [1]:#分子的权重设置为0.1和1
    for w2 in [1]:#分母的权重设置为
            for w4 in [10,50]:

                os.system (f'python /home/lym_pcl/lym_dataset/code_wholenet/code_wholenet_losspara_optima/mulit_branch_with_fenzifenmu/main_train.py\
                           --w1 {w1} --w2 {w2} --w4 {w4} \
                            --batch_size 8 --epochs 200  \
                            --wandb_project_name "Nguyen_1523"\
                            --root "/home/lym_pcl/lym_dataset/Nguyen_1523/" \
                            --model "Result/WHOLE_NET/4_9_withfenzi_fenmu2"')