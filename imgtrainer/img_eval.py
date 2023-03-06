from img_set import NameImgSet
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


def compute_result(csv_path,output_path):
    txt_writer = open(output_path,'w')
    csv_reader = csv.reader(open(csv_path,'r')) 
    all_pred = {}
    all_label = {}
    
    preds = []
    labels = []
    for row in islice(csv_reader,1,None):
        name = row[0]
       
        pid = name.split('_')[0]
        
        
    
        label = row[1]
        pred = row[5]
        

        preds.append(float(pred))
        labels.append(int(label))
        
        if pid not in all_pred.keys():
            all_pred[pid] = [float(pred)]
            all_label[pid] = int(label)
        else:
            all_pred[pid].append(float(pred))
    
    all_pred_list = []
    all_label_list = []
    
    for pid in all_pred.keys():
        all_pred_list.append(np.mean(all_pred[pid]))
        all_label_list.append(all_label[pid])
        
    
    y = np.array(labels)
    pred = np.array(preds)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    acc = metrics.accuracy_score(y, pred > 0.5 )
    auc = metrics.auc(fpr, tpr) 
    
    msg = 'Image Acc: {} '.format(acc)
    txt_writer.write(msg + '\n')
    txt_writer.flush()
    
    msg = 'Image AUC: {} '.format(auc)
    txt_writer.write(msg + '\n')
    txt_writer.flush()
    
    ######
    
    y = np.array(all_label_list)
    pred = np.array(all_pred_list)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    acc = metrics.accuracy_score(y, pred > 0.5 )
    auc = metrics.auc(fpr, tpr) 
    
    msg = 'Patient Acc: {} '.format(acc)
    txt_writer.write(msg + '\n')
    txt_writer.flush()
    
    msg = 'Patient AUC: {} '.format(auc)
    txt_writer.write(msg + '\n')
    txt_writer.flush()
    
    
    
    
class ImgBinaryEvaluater:
    def __init__(
        self,
        input_path:str,
        gpu_id: int,
        name: Optional[str] = None,
        data_path: Optional[str] = None,
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
        
        testset = NameImgSet(
            root=config.data.test_root if data_path is None else data_path,
            transform=test_transform,
        )
        
        test_loader = DataLoader(
            dataset = testset,
            batch_size = config.env.batch_size,
            shuffle = False,
            num_workers = config.env.num_workers,
        )
        
        self.test_loader = test_loader 
        
       
        self.output_path = input_path
        
        model = create_model(
            config = config.models,
            num_classes = config.data.num_classes,
        )
        
        model.load_state_dict(
            torch.load(os.path.join(input_path,'weights',choice,'best_model.pth'))
        )
        
        self.name = name
        self.model = model.to(self.device)
        print('Initalization complete.')
        
        
    def evaluate(self):
        if self.name is None:
            csv_path = os.path.join(self.output_path,'prediction_{}.csv'.format(self.choice))
            txt_path = os.path.join(self.output_path,'patient_result_{}.txt'.format(self.choice))
        else:
            csv_path = os.path.join(self.output_path,'{}_prediction_{}.csv'.format(self.name,self.choice))
            txt_path = os.path.join(self.output_path,'{}_patient_result_{}.txt'.format(self.name,self.choice))
        f = open(csv_path,'w',encoding='utf-8')
        
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ['name','label','logits_0','logits_1','prob_0','prob_1']
        )
        

        with torch.no_grad():
            self.model.eval()
            
            for img, label, img_path  in tqdm(self.test_loader):

                
                img = img.to(self.device)
                label = label.to(self.device)
                
                data = {}
                for idx, model_prefix in enumerate(self.model_prefixs):
                    data[f"{model_prefix}_{IMAGE}"] = img
                

                out = self.model(data)[self.model.prefix][LOGITS]
                
                out = out.cpu().numpy()
                label = label.cpu().tolist()
                img_name = [name.split('/')[-1] for name in img_path]
                
                for idx in range(len(img_name)):
                    logits = np.array(out[idx,:].tolist())
                    probs = np.exp(logits) / np.sum(np.exp(logits))
            
                    result = [
                        img_name[idx],
                        label[idx],]
                    
                    result = result + list(logits) + list(probs)
                    
                    csv_writer.writerow(result)
                
        compute_result(csv_path,txt_path)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default='/home/wang1/shenyiqing/Results/ImgTrain/may10/pdl1_v11_set3')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--choice', type=str, default='auc')
    args = parser.parse_args()  
    
    evaluter = ImgBinaryEvaluater(
        input_path=args.path,
        data_path=args.data_path,
        name=args.name,
        gpu_id=args.gpu_id,
        choice=args.choice,
    )
    evaluter.evaluate()
