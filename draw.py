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

# 5 robots
# smooth(read_path='./5_robots_data_for_figure/', save_path='./5_robots_data_for_figure/', file_name='0711-11-30.csv')
# smooth(read_path='./5_robots_data_for_figure/', save_path='./5_robots_data_for_figure/', file_name='0712-09-05.csv')
# smooth(read_path='./5_robots_data_for_figure/', save_path='./5_robots_data_for_figure/', file_name='0712-20-12.csv')
# smooth(read_path='./5_robots_data_for_figure/', save_path='./5_robots_data_for_figure/', file_name='0722-09-48.csv')
# smooth(read_path='./5_robots_data_for_figure/', save_path='./5_robots_data_for_figure/', file_name='0724-09-14.csv')
#
# df1 = pd.read_csv('./5_robots_data_for_figure/smooth_0711-11-30.csv')
# df2 = pd.read_csv('./5_robots_data_for_figure/smooth_0712-09-05.csv')
# df3 = pd.read_csv('./5_robots_data_for_figure/smooth_0712-20-12.csv')
# df4 = pd.read_csv('./5_robots_data_for_figure/smooth_0722-09-48.csv')
# df5 = pd.read_csv('./5_robots_data_for_figure/smooth_0724-09-14.csv')
#
# df = df1.append(df2.append(df3.append(df4.append(df5))))
#
# plt.figure(figsize=(15, 10))
# fig = sns.lineplot(data=df, x="Episodes", y="Rewards", color="Blue")
# fig.tick_params(axis='x', labelsize=25)
# fig.tick_params(axis='y', labelsize=25)
# fig.set_xlabel('Episodes', fontsize=30)
# fig.set_ylabel('Rewards', fontsize=30)
# plt.title("Training Rewards for 5 Robots", fontsize=30)



# 10
smooth(read_path='./10_robots_data_for_figure/', save_path='./10_robots_data_for_figure/', file_name='0115-09-31.csv')
smooth(read_path='./10_robots_data_for_figure/', save_path='./10_robots_data_for_figure/', file_name='0115-19-15.csv')
smooth(read_path='./10_robots_data_for_figure/', save_path='./10_robots_data_for_figure/', file_name='0116-14-58.csv')
# smooth(read_path='./10_robots_data_for_figure/', save_path='./10_robots_data_for_figure/', file_name='0725-09-25.csv')

df1 = pd.read_csv('./10_robots_data_for_figure/smooth_0115-09-31.csv')
df2 = pd.read_csv('./10_robots_data_for_figure/smooth_0115-19-15.csv')
df3 = pd.read_csv('./10_robots_data_for_figure/smooth_0116-14-58.csv')
# df4 = pd.read_csv('./10_robots_data_for_figure/smooth_0725-09-25.csv')

df = pd.concat([df1, df2, df3], ignore_index=True)

plt.figure(figsize=(15, 10))
fig = sns.lineplot(data=df, x="Episodes", y="Rewards", color="Blue")
fig.tick_params(axis='x', labelsize=25)
fig.tick_params(axis='y', labelsize=25)
fig.set_xlabel('Episodes', fontsize=40)
fig.set_ylabel('Rewards', fontsize=40)
plt.title("Training Rewards for 10 Robots", fontsize=40)

from matplotlib.patches import Patch
mean_patch = plt.Line2D([0], [0], color='blue', lw=2, label='Mean')
std_dev_patch = Patch(facecolor="blue", edgecolor='blue', label='Standard Deviation', alpha=0.2)

plt.legend(handles=[mean_patch, std_dev_patch], loc='upper right', bbox_to_anchor=(1, 0.4), borderpad=0.5, fontsize=18)

plt.show()
