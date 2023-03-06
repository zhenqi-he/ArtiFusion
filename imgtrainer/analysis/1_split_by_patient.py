import csv
from itertools import islice
import numpy as np
import os
csv_path = '/home/wang1/shenyiqing/Results/ImgTrain/may10/pdl1_v12_set3/prediction_acc.csv'
out_path = '/home/wang1/shenyiqing/Results/ImgTrain/may10/pdl1_v12_set3/prediction_acc'

os.makedirs(out_path,exist_ok=True)

csv_reader = csv.reader(open(csv_path,'r')) 


all_pred = {}
all_img_name = {}
all_label = {}

for row in islice(csv_reader,1,None):
    name = row[0]
    
    pid = name.split('_')[0]
    
    label = row[1]
    pred = row[5]
    
    if pid not in all_pred.keys():
        all_pred[pid] = [float(pred)]
        all_label[pid] = int(label)
        all_img_name[pid] = [row[0]]
    else:
        all_pred[pid].append(float(pred))
        all_img_name[pid].append(row[0])
    
all_pred_list = []
all_label_list = []

for pid in all_pred.keys():
    
    f = open(os.path.join(out_path,'{}.csv'.format(pid)),'w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['image_name', 'x' , 'y' , 'label','pred'])
    
    label = all_label[pid]
    
    for img_name, pred in zip(all_img_name[pid],all_pred[pid]):
        
        x = img_name.split('_')[-2]
        y = (img_name.split('_')[-1]).split('.')[0]
        
        csv_writer.writerow([img_name,x,y,label,pred])

print('Complete')