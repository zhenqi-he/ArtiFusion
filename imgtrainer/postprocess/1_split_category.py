import os 
import csv 
from itertools import islice
from shutil import copyfile
from tqdm import tqdm 


img_path = '/home/wang1/shenyiqing/Results/stad_rois/stad_binary1_train/1'
out_path= '/home/wang1/shenyiqing/dataset/stad_rois_split/stad_binary1_train'

for label in ['0','1']:
    os.makedirs(os.path.join(out_path,label),exist_ok=True)

csv_path = '/home/wang1/shenyiqing/imgtrainer/postprocess/pdl1_label.csv'
csv_reader =  csv.reader(open(csv_path,'r')) 

label_dict = {}
for row in islice(csv_reader,1,None):
    label_dict[row[0]]=row[1]

for img in tqdm(os.listdir(img_path)):
    pid = img.split('_')[0]
    pid = pid.split('-')[-1]
    label = label_dict[pid]
    
    copyfile(
        os.path.join(img_path,img),
        os.path.join(out_path,label,img)
    )

