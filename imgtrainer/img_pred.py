from img_set import NameImgSet,NameImgPredSet
from img_model import create_model 
import os
import torch 
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import lr_scheduler 
from typing import Optional
import time 
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random 
from shutil import copyfile
from omegaconf import OmegaConf
from torchvision import transforms
import csv 
from itertools import islice
from sklearn import metrics
from utils import (
    AverageMeter,
    AUCRecorder,
    accuracy,
) 
from utils import(
    IMAGE,
    LABEL,
    LOGITS, 
    FEATURES,
    TIMM_MODEL,
    FUSION_MLP
)



train_transform=transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])



    
class ImgBinaryPredicter:
    def __init__(
        self,
        input_path:str,
        data_path:str,
        output_path:str,
        gpu_id: int,
        choice: Optional[str] = 'acc',
    ):
        config = OmegaConf.load(os.path.join(input_path,'config.yaml'))
        
        self.choice = choice
        if config.env.seed is not None:
            torch.manual_seed(config.env.seed)
            np.random.seed(config.env.seed)
            random.seed(config.env.seed)
        
        self.model_prefixs = []
        for mode_name in config.models.names:
            if mode_name.lower().startswith(TIMM_MODEL):
                self.model_prefixs.append(mode_name)
                
        self.device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
        
        testset = NameImgPredSet(
            root=data_path,
            transform=test_transform,
        )
        
        test_loader = DataLoader(
            dataset = testset,
            batch_size = config.env.batch_size,
            shuffle = False,
            num_workers = config.env.num_workers,
        )
        
        self.test_loader = test_loader 
       
        self.output_path = output_path
        os.makedirs(self.output_path,exist_ok=True)
        
        model = create_model(
            config = config.models,
            num_classes = config.data.num_classes,
        )
        
        model.load_state_dict(
            torch.load(os.path.join(input_path,'weights',choice,'best_model.pth'))
        )
        
        self.model = model.to(self.device)
        
        self.data_path = data_path
        
        print('Initalization complete.')
        
        
    def evalute(self):
        csv_path = os.path.join(self.output_path,'prediction_{}.csv'.format(self.choice))
        f = open(csv_path,'w',encoding='utf-8')
        
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ['name','logits_0','logits_1','prob_0','prob_1']
        )
        

        with torch.no_grad():
            self.model.eval()
            
            for img, img_path in tqdm(self.test_loader):

                
                img = img.to(self.device)
                
                data = {}
                for idx, model_prefix in enumerate(self.model_prefixs):
                    data[f"{model_prefix}_{IMAGE}"] = img
                

                out = self.model(data)[self.model.prefix][LOGITS]
                
                out = out.cpu().numpy()
               
                img_name = [name.split('/')[-1] for name in img_path]
                
                for idx in range(len(img_name)):
                    logits = np.array(out[idx,:].tolist())
                    probs = np.exp(logits) / np.sum(np.exp(logits))
            
                    result = [img_name[idx]]
                    
                    result = result + list(logits) + list(probs)
                    
                    csv_writer.writerow(result)
                    
    def evalute_cam(self):
        
        from torchcam.methods import SmoothGradCAMpp
        from torchcam.utils import overlay_mask
        
        from PIL import Image

        from torchvision.transforms.functional import normalize, resize, to_pil_image
        
        import matplotlib.pyplot as plt
        self.model.eval()
        
        cam_extractor = SmoothGradCAMpp(self.model.model)
        
        for img, img_path in tqdm(self.test_loader):

            
            img = img.to(self.device)
            
            data = {}
            for idx, model_prefix in enumerate(self.model_prefixs):
                data[f"{model_prefix}_{IMAGE}"] = img
            

            out = self.model(data)[self.model.prefix][LOGITS]
            
            img_name = [name.split('/')[-1] for name in img_path]
            
            for i in range(out.shape[0]):
                
                activation_map = cam_extractor(
                    out[i].squeeze(0).argmax().item(), 
                    out[i].unsqueeze(0)
                )
                
                mask = activation_map[0].squeeze(0).cpu().numpy()[i]
                

                mask = overlay_mask(
                    # to_pil_image(Image.open(os.path.join(self.data_path,img_name[i]))), 
                    Image.open(os.path.join(self.data_path,img_name[i])).convert("RGB"), 
                    to_pil_image(mask, mode='F'), 
                    alpha=0.5
                )
                
                plt.imshow(mask)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path,img_name[i]))
                   
                

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default='/home/yiqing/data/mri_bone/result')
    parser.add_argument('--data_path', type=str, default='/home/yiqing/data/mri_bone/img/t1c_bone_test')
    parser.add_argument('--out_path', type=str, default='/home/yiqing/data/mri_bone/result/out/t1c_bone_test')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--choice', type=str, default='acc')
    args = parser.parse_args()  
    
    evaluter = ImgBinaryPredicter(
        input_path=args.path,
        data_path=args.data_path,
        output_path=args.out_path,
        gpu_id=args.gpu_id,
        choice=args.choice,
    )
    evaluter.evalute_cam()
