import os 
from tqdm import tqdm
import cv2 

size = (224,224)
input_dir = '/home/wang1/shenyiqing/dataset/512_gist'
output_dir = '/home/wang1/shenyiqing/dataset/224_gist'

total_num = len(os.listdir(input_dir))
for idx, name in enumerate(os.listdir(input_dir)):
    os.makedirs(os.path.join(output_dir,name),exist_ok=True)
    
    print("Start processing {} of {} ".format(idx+1,total_num))
    for img_name in tqdm(os.listdir(os.path.join(input_dir,name))):
        img = cv2.imread(os.path.join(input_dir,name,img_name))
        img_r = cv2.resize(img,size)
        cv2.imwrite(os.path.join(output_dir,name,img_name),img_r)


