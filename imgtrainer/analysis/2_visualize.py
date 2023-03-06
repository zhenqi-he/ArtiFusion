
import csv
from itertools import islice
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


in_path = '/home/wang1/shenyiqing/Results/ImgTrain/may10/pdl1_v12_set3/prediction_acc'
out_path = in_path + '_visual'
os.makedirs(out_path,exist_ok=True)

for name in os.listdir(in_path):
    
    csv_reader = csv.reader(open(os.path.join(in_path,name),'r'))
    
    xs = []
    ys = []
    for row in islice(csv_reader,1,None):
        x = int(row[1])
        y = int(row[2])
        label = row[3]
        xs.append(x)
        ys.append(y)
        
    
    xmax = max(xs)
    xmin = min(xs)
    ymax = max(ys)
    ymin = min(ys)

    delta_x = xmax - xmin + 1
    delta_y = ymax - ymin + 1
    
    heatmap = np.zeros((delta_x,delta_y))
    
    csv_reader = csv.reader(open(os.path.join(in_path,name),'r'))
    for row in islice(csv_reader,1,None):
        x = int(row[1]) - xmin
        y = int(row[2]) - ymin
        score = float(row[4]) * 2 - 1
        
        
        heatmap[x,y] = score
        
        
    sns.set()
    fig = plt.figure()
    sns.color_palette("vlag", as_cmap=True)
    sns_plot = sns.heatmap(heatmap, vmin=-1, vmax=1, cmap='bwr')
    sns_plot.tick_params(left=False, bottom=False)
    cbar = sns_plot.collections[0].colorbar
    cbar.set_ticks([-1, -.5, 0, .5, 1])
    img_name = name.split('.')[0]
    plt.savefig(os.path.join(out_path,f"{img_name}_{label}.png"))
    plt.close()
    # break
    
    
    
    
