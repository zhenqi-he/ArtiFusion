import os 
import csv 
from itertools import islice
from shutil import copyfile
from tqdm import tqdm 

csv_path = '/home/wang1/shenyiqing/Results/ImgTrain/may10/stad_binary3/train_pred/prediction_acc.csv'
img_path = '/home/wang1/shenyiqing/dataset/224_gist_split_whole/train'

out_path = '/home/wang1/shenyiqing/Results/stad_rois/stad_binary3_train'

for label in ['0','1']:
    os.makedirs(os.path.join(out_path,label),exist_ok=True)

csv_reader = csv.reader(open(csv_path,'r')) 
for row in tqdm(islice(csv_reader,1,None)):
    
    if float(row[4]) > float(row[3]) :
        label = '1'
    else:
        label = '0'
    
    src_img_path = os.path.join(img_path,row[0])
    dst_img_path = os.path.join(out_path,label,row[0])
    copyfile(src_img_path,dst_img_path)