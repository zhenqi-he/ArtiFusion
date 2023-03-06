from img_set import ImgSet
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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# To fix the EOFError,discribed in https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0

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


class ImgTrainer:
    def __init__(
        self,
        config_path:str,
    ):
        config = OmegaConf.load(config_path)
        
        if hasattr(config.env, "seed"):
            torch.manual_seed(config.env.seed)
            np.random.seed(config.env.seed)
            random.seed(config.env.seed)
            
        self.device = torch.device('cuda:{}'.format(config.env.gpu_id) if torch.cuda.is_available() else 'cpu')
        
        
        train_transforms = []
        self.model_prefixs = []
        
        drop_last=False
        for model_name in config.models.names:
            if not model_name.lower().startswith(FUSION_MLP):
                self.model_prefixs.append(model_name)
                train_transforms.append(train_transform)

            if model_name.lower().startswith('mem_vit'):
                drop_last=True
                
        trainset = ImgSet(
            root=config.data.train_root,
            transform_list=train_transforms, 
        )
        
        train_loader = DataLoader(
            dataset = trainset,
            batch_size = config.env.batch_size,
            shuffle = True,
            pin_memory=False,
            num_workers = config.env.num_workers,
            drop_last=drop_last,
        )
        
        testset = ImgSet(
            root=config.data.test_root,
            transform=test_transform,
        )
        
        test_loader = DataLoader(
            dataset = testset,
            batch_size = config.env.batch_size,
            shuffle = True,
            num_workers = config.env.num_workers,
            drop_last=drop_last,
        )
        
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        
        self.output_path = config.output.path
      
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path,'weights','acc'), exist_ok=True)
        
        if config.data.num_classes == 2:
            os.makedirs(os.path.join(self.output_path,'weights','auc'), exist_ok=True)
            os.makedirs(os.path.join(self.output_path,'figures'), exist_ok=True)
        
    
        self.logging = open(os.path.join(self.output_path,'logging.txt'), 'w+')
        copyfile(config_path,os.path.join(self.output_path,'config.yaml'))
        
        
        model = create_model(
            config = config.models,
            num_classes = config.data.num_classes,
        )
        
        self.model = model.to(self.device)
        
        
        self.optimizer = AdamW(
            params = self.model.parameters(), 
            lr = config.opt.learning_rate, 
            weight_decay = config.opt.weight_decay
        ) 
        
        
        if config.opt.scheduler.lower() == 'epoential':
            self.scheduler = lr_scheduler.ExponentialLR(
                optimizer = self.optimizer, 
                gamma = config.opt.gamma
            )
        elif config.opt.scheduler.lower() == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer = self.optimizer, 
                T_max = 10,
                eta_min = config.opt.min_learning_rate,
            )
        elif config.opt.scheduler.lower() == 'constant':
            self.scheduler = lr_scheduler.ConstantLR(
                optimizer = self.optimizer, 
            )
        else:
            raise ValueError("Unkown scheduler {}".format(config.opt.scheduler.lower()))
        self.epochs = config.opt.epochs
        self.patience = config.opt.patience
        
        self.is_binary = config.data.num_classes == 2
        
    def train(
        self,
        verbosity: Optional[bool] = True,
    ):
        if self.is_binary:
            self.binary_train(verbosity)
        else:
            self.multi_train(verbosity)

    def binary_train(
        self,
        verbosity: Optional[bool] = True,
    ):
        best_test_acc = 0.0
        best_test_auc = 0.0
        best_epoch = 0
        
        time_start=time.time()
        
        msg = 'Total training epochs : {}\n'.format(self.epochs) 
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
            
        for epoch in range(1,self.epochs+1):
            train_loss, train_acc, train_auc = self._binary_train_one_epoch()
            test_loss, test_acc, test_auc, _test_auc_recorder = self._binary_test_per_epoch(model=self.model)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc 
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'acc' , 'model_epoch{}.pth'.format(epoch)))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'acc' , 'best_model.pth'))
                
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_auc_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'auc' , 'model_epoch{}.pth'.format(epoch)))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'auc' , 'best_model.pth'))
                
                _test_auc_recorder.draw_roc(
                    path = os.path.join(self.output_path,'figures','epoch_{}_test_roc.png'.format(epoch))
                )
            
            msg = 'Epoch {:03d} ##################   \
                \n \tTrain loss: {:.5f},   Train acc: {:.3f}%,    Train auc: {:.4f};\
                \n \tTest loss: {:.5f},   Test acc: {:.3f}%,   Test auc: {:.4f};  \
                \n \tBest test acc: {:.3f},    Best test auc: {:.4f}\n\n'.format(
                    epoch, train_loss, train_acc, train_auc,
                    test_loss, test_acc, test_auc, 
                    best_test_acc, best_test_auc)  
                
            if verbosity:
                print(msg)
            self.logging.write(msg)
            self.logging.flush()   
            
            if (epoch - best_epoch) > self.patience:
                break
        
        msg = "Best test acc:{:.3f}% @ epoch {} \n".format(best_test_acc,best_epoch)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
        
        msg = "Best test auc:{:.4f} @ epoch {} \n".format(best_test_auc,best_auc_epoch)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
        
        
        time_end=time.time()    
        msg= "run time: {:.1f}s, {:.2f}h\n".format(time_end-time_start,(time_end-time_start)/3600)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 

    
    def _binary_train_one_epoch(self):  
        _train_loss_recorder = AverageMeter()
        _train_acc_recorder = AverageMeter()
        _train_auc_recorder = AUCRecorder()
        
        self.model.train()
        
        for img, label in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            
            img = img.to(self.device)
            label = label.to(self.device)
            
            data = {}
            for idx, model_prefix in enumerate(self.model_prefixs):
                data[f"{model_prefix}_{IMAGE}"] = img[:,idx, ...]
            
            out = self.model(data)[self.model.prefix][LOGITS]
            loss = F.cross_entropy(out,label)
            
            loss.backward() 
            self.optimizer.step()
                
            acc = accuracy(out, label)[0]
            _train_loss_recorder.update(loss.item(), out.size(0))
            _train_acc_recorder.update(acc.item(), out.size(0))
            _train_auc_recorder.update(prediction=out[:,1],target=label)

        self.scheduler.step()
        
        train_loss = _train_loss_recorder.avg 
        train_acc = _train_acc_recorder.avg 
        train_auc = _train_auc_recorder.auc
        
        return train_loss, train_acc, train_auc
    
    def _binary_test_per_epoch(self, model):
        _test_loss_recorder = AverageMeter()
        _test_acc_recorder = AverageMeter()
        _test_auc_recorder = AUCRecorder()
        
        with torch.no_grad():
            model.eval()
            
            for img, label in tqdm(self.test_loader):
                
                img = img.to(self.device)
                label = label.to(self.device)
                
                data = {}
                for _ , model_prefix in enumerate(self.model_prefixs):
                    data[f"{model_prefix}_{IMAGE}"] = img
            
                out  = model(data)[self.model.prefix][LOGITS]
                
                loss = F.cross_entropy(out,label)
                
                acc = accuracy(out, label)[0]

                _test_loss_recorder.update(loss.item(), out.size(0))
                _test_acc_recorder.update(acc.item(), out.size(0))
                _test_auc_recorder.update(prediction=out[:,1],target=label)
        
        test_loss = _test_loss_recorder.avg 
        test_acc = _test_acc_recorder.avg 
        test_auc = _test_auc_recorder.auc
        
        
        return test_loss, test_acc, test_auc, _test_auc_recorder
    
    def multi_train(
        self,
        verbosity: Optional[bool] = True,
    ):
        best_epoch = 0
        best_test_acc = 0 
        
        time_start=time.time()
        
        msg = 'Total training epochs : {}\n'.format(self.epochs) 
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
            
        for epoch in range(1,self.epochs+1):
            train_loss, train_acc = self._multi_train_one_epoch()
            test_loss, test_acc = self._multi_test_per_epoch(model=self.model)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc 
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'acc' , 'model_epoch{}.pth'.format(epoch)))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'acc' , 'best_model.pth'))
                
            
            
            msg = 'Epoch {:03d} ##################   \
                \n \tTrain loss: {:.5f},   Train acc: {:.3f}%;\
                \n \tTest loss: {:.5f},   Test acc: {:.3f}%;  \
                \n \tBest test acc: {:.3f}\n\n'.format(
                    epoch, train_loss, train_acc,
                    test_loss, test_acc, 
                    best_test_acc)  
                
            if verbosity:
                print(msg)
            self.logging.write(msg)
            self.logging.flush()   
            
            if (epoch - best_epoch) > self.patience:
                break
        
        msg = "Best test acc:{:.3f}% @ epoch {} \n".format(best_test_acc,best_epoch)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
        
        time_end=time.time()    
        msg= "run time: {:.1f}s, {:.2f}h\n".format(time_end-time_start,(time_end-time_start)/3600)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush()
         
    def _multi_train_one_epoch(self):  
        _train_loss_recorder = AverageMeter()
        _train_acc_recorder = AverageMeter()
        
        self.model.train()
        
        for img, label in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            
            img = img.to(self.device)
            label = label.to(self.device)
            
            data = {}
            for idx, model_prefix in enumerate(self.model_prefixs):
                data[f"{model_prefix}_{IMAGE}"] = img[:,idx, ...]
            
            out = self.model(data)[self.model.prefix][LOGITS]
            loss = F.cross_entropy(out,label)
            
            loss.backward() 
            self.optimizer.step()
                
            acc = accuracy(out, label)[0]
            _train_loss_recorder.update(loss.item(), out.size(0))
            _train_acc_recorder.update(acc.item(), out.size(0))

        self.scheduler.step()
        
        train_loss = _train_loss_recorder.avg 
        train_acc = _train_acc_recorder.avg 
        
        return train_loss, train_acc
    
    
    def _multi_test_per_epoch(self, model):
        _test_loss_recorder = AverageMeter()
        _test_acc_recorder = AverageMeter()
        
        with torch.no_grad():
            model.eval()
            
            for img, label in tqdm(self.test_loader):
                
                img = img.to(self.device)
                label = label.to(self.device)
                
                data = {}
                for _ , model_prefix in enumerate(self.model_prefixs):
                    data[f"{model_prefix}_{IMAGE}"] = img
            
                out  = model(data)[self.model.prefix][LOGITS]
                
                loss = F.cross_entropy(out,label)
                
                acc = accuracy(out, label)[0]

                _test_loss_recorder.update(loss.item(), out.size(0))
                _test_acc_recorder.update(acc.item(), out.size(0))

        test_loss = _test_loss_recorder.avg 
        test_acc = _test_acc_recorder.avg 

        
        return test_loss, test_acc
