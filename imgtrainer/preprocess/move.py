import os 
from shutil import copyfile
from tqdm import tqdm

input_dir = '/home/wang1/shenyiqing/dataset/224_gist'
output_dir = '/home/wang1/shenyiqing/dataset/224_gist_split'



for name in tqdm(os.listdir(input_dir)):
    if 's' in name:
        middle = 'train'
    else:
        middle = 'test'
        
    os.makedirs(os.path.join(output_dir,middle,name),exist_ok=True)
    
    for img_name in os.listdir(os.path.join(input_dir,name)):
        copyfile(
            os.path.join(input_dir,name,img_name),
            os.path.join(output_dir,middle,name,img_name)
        )