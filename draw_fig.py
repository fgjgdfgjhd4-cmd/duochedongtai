"""这个文件用来画训练reward图"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def smooth(read_path, save_path, file_name, x='Episodes', y='Rewards', weight=0.75):
    data = pd.read_csv(read_path + file_name)
    scalar = data[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({x: data[x].values, y: smoothed})
    save.to_csv(save_path + 'smooth_' + file_name)

smooth(read_path='./5_robots_data_for_figure', save_path='./5_robots_data_for_figure', file_name='0711-11-30.csv')
smooth(read_path='./5_robots_data_for_figure', save_path='./5_robots_data_for_figure', file_name='0712-09-05.csv')
smooth(read_path='./5_robots_data_for_figure', save_path='./5_robots_data_for_figure', file_name='0712-20-12.csv')
smooth(read_path='./5_robots_data_for_figure', save_path='./5_robots_data_for_figure', file_name='0722-09-48.csv')
smooth(read_path='./5_robots_data_for_figure', save_path='./5_robots_data_for_figure', file_name='0724-09-14.csv')

df1 = pd.read_csv('./5_robots_data_for_figure/smooth_0711-11-30.csv')

plt.figure(figsize=(15, 10))
sns.lineplot(data=df1, x="Episodes", y="Rewards")

