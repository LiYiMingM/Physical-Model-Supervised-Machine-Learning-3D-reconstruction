# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:53:58 2023

@author: 34677
"""

import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
# /data/lym/lymfile/dataset/real_dataset/whole_process_914/dataset_914/Result/WHOLE_NET/10.15_1/train_loss and others.csv
path1='/data/liuhee/whole_process_914/dataset_914'
path2='/Result/WHOLE_NET/10_19_2000_2/test_result/'
filename = pd.read_csv(path1+path2+'test_loss_unwrap.csv')
l1_loss = filename['loss1_unwrap_ave']

plt.scatter(range(len(l1_loss)), l1_loss)
plt.ylim(0, max(l1_loss)) #y轴坐标范围，参考模型具体效果调整范围
plt.xlabel('Data Point')
plt.ylabel('L1 Loss')
plt.title('Scatter Plot of L1 Loss')
plt.show()
rmse_loss = filename['rmse_unwrap_loss']

plt.scatter(range(len(rmse_loss)), rmse_loss)
plt.ylim(0, max(rmse_loss)) #y轴坐标范围，参考模型具体效果调整范围
plt.xlabel('Data Point')
plt.ylabel('Rmse Loss')
plt.title('Scatter Plot of rmse Loss')
plt.show()

# 提取 l1_loss 列的数据
data = filename['loss1_unwrap_ave'].values.astype(float)

# 计算标准差
std_deviation = np.std(data)

# 更新 DataFrame 中的 sd 列


print("标准差:", std_deviation)
filename['sd'] = std_deviation
# 将 DataFrame 保存回 CSV 文件
filename.to_csv(path1+path2+' test_loss_unwrap.csv', index=False)

print("标准差已写入CSV文件.")