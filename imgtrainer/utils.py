from torch import nn
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt 

# input
IMAGE = "image"
LABEL = "label"

# output
LOGITS = "logits"
FEATURES = "features"

# model
TIMM_MODEL = 'timm'
FUSION_MLP = 'fusion_mlp'

def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
        


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AUCRecorder(object):
    def __init__(self):
        self.prediction = []
        self.target = []
        
    def update(self,prediction,target):
        self.prediction = self.prediction + prediction.tolist()
        self.target = self.target + target.tolist()
    
    @property
    def auc(self):
        prediction = np.array(self.prediction)
        target = np.array(self.target)
        fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=1)
        auc = metrics.auc(fpr, tpr)  
        return auc
    
    def draw_roc(self, path):
        prediction = np.array(self.prediction)
        target = np.array(self.target)
        fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=1)
        auc = metrics.auc(fpr, tpr)  
        
        plt.figure(figsize=(4.5,4.5))

        x = np.arange(0,1.01,0.01)

        plt.plot(x,x,ls="--",color='grey',alpha=0.5)
        plt.plot(fpr,tpr,label='auc {:.3f}'.format(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(path,dpi=300)

