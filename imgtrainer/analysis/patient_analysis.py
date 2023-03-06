import csv
from itertools import islice
import numpy as np

csv_path = '/home/wang1/shenyiqing/Results/ImgTrain/may10/pdl1_v10_set3/test_prediction_auc.csv'
out_path = '/home/wang1/shenyiqing/Results/ImgTrain/may10/pdl1_v10_set3/test_prediction_auc_patient.csv'

print(out_path)

csv_reader = csv.reader(open(csv_path,'r')) 
f = open(out_path,'w',encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(['pid','label','pred'])

all_pred = {}
all_label = {}

for row in islice(csv_reader,1,None):
    name = row[0]
    
    pid = name.split('_')[0]
    
    label = row[1]
    pred = row[5]
    
    if pid not in all_pred.keys():
        all_pred[pid] = [float(pred)]
        all_label[pid] = int(label)
    else:
        all_pred[pid].append(float(pred))
    
all_pred_list = []
all_label_list = []

for pid in all_pred.keys():
    pred = np.mean(all_pred[pid])
    label = all_label[pid]
    csv_writer.writerow([pid,label,pred])

print('Complete')