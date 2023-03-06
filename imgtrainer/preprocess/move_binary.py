import os
from shutil import copyfile 


in_dir = '/home/wang1/shenyiqing/dataset/STAD/'
out_dir = '/home/wang1/shenyiqing/dataset/STAD_binary'

for mode in os.listdir(in_dir):
    for category in os.listdir(os.path.join(in_dir,mode)):
        if category in ['1','2','4'] : 
            label = '1'
        else:
            label = '0'
        out = os.path.join(out_dir,mode,label)
        os.makedirs(out,exist_ok=True)
        for img_name in os.listdir(os.path.join(in_dir,mode,category)):
            copyfile(
                os.path.join(in_dir,mode,category,img_name),
                os.path.join(out,img_name)
            )