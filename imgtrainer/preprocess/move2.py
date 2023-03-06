import os 
from shutil import copyfile
from tqdm import tqdm


input_dir = '/home/wang1/shenyiqing/dataset/224_gist_split'
output_dir = '/home/wang1/shenyiqing/dataset/224_gist_split_whole'

for mode in os.listdir(input_dir):
    os.makedirs(os.path.join(output_dir,mode),exist_ok=True)
    
    for name in tqdm(os.listdir(os.path.join(input_dir,mode))):
        for img_name in os.listdir(os.path.join(input_dir,mode,name)):
            copyfile(
                os.path.join(input_dir,mode,name,img_name),
                os.path.join(output_dir,mode,img_name),
            )


