import timm
import torch
from torch import nn
from typing import List, Optional, Tuple
from utils import init_weights
from utils import(
    IMAGE,
    LABEL,
    LOGITS, 
    FEATURES,
    TIMM_MODEL,
    FUSION_MLP
)
from model.mlp import MLP
from model.mem_vit import MemorizingTransformer
from model.prenorm import LabNorm
import functools

class TIMM(nn.Module):
    def __init__(
            self,
            prefix: str,
            model_name: str,
            num_classes: Optional[int] = 0,
            pretrained: Optional[bool] = True,
            prenorm: Optional[bool] = False,
    ):
        super(TIMM, self).__init__()
        self.prefix = prefix 
        self.data_key = f"{prefix}_{IMAGE}"
        self.label_key = f"{LABEL}"
        
        self.num_classes = num_classes
        
        self.normlize = LabNorm() if prenorm else nn.Identity()
        
        if prenorm:
            print('Using Prenorm scheme.')
        
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        
        self.out_features = self.model.num_features
        
        self.head = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(init_weights)
        
     
    def forward(
        self, 
        batch
    ):
        data = batch[self.data_key]
        data = self.normlize(data)
        features = self.model(data)
        logits = self.head(features)
            
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }     
        

class MEMORYVIT(nn.Module):
    def __init__(
            self,
            prefix: str,
            num_classes: Optional[int] = 0,
            batch_size=32,
    ):
        super(MEMORYVIT, self).__init__()
        self.prefix = prefix 
        self.data_key = f"{prefix}_{IMAGE}"
        self.label_key = f"{LABEL}"
        
        self.num_classes = num_classes
        
        self.model = MemorizingTransformer(
                image_size=224,
                num_tokens=128,  # number of tokens
                dim=128,  # dimension
                dim_head=128,  # dimension per attention head
                depth=7,  # number of layers
                memorizing_layers=(
                    7 - 1,
                ),  # which layers to have ANN memories
                max_knn_memories=8192,  # maximum ANN memories to keep (once it hits this capacity, it will be reset for now, due to limitations in faiss' ability to remove entries)
                num_retrieved_memories=32,  # number of ANN memories to retrieve
                # num_classes=num_classes,  # number of classes
                global_pool=True,
            )
        
        self.knn_memories = self.model.create_knn_memories(batch_size=batch_size)
        
        self.out_features = 128
        
        self.head = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(init_weights)
        
     
    def forward(
        self, 
        batch
    ):
        data = batch[self.data_key]
        # print(data.shape)
        features = self.model(data, self.knn_memories)
        logits = self.head(features)
            
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }     
        

class FusionMLP(nn.Module):
    def __init__(
        self,
        prefix: str,
        models: list,
        num_classes: int,
        hidden_features: List[int],
        adapt_in_features: Optional[str] = None,
        activation: Optional[str] = "gelu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
    ):
    
        super().__init__()
        self.prefix = prefix
        self.model = nn.ModuleList(models)   
        

        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )
            
            in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in range(len(raw_in_features))]
            )
            in_features = sum(raw_in_features)

        assert len(self.adapter) == len(self.model)
        
        
        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    normalization=normalization,
                )
            )
            in_features = per_hidden_features
            
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        # in_features has become the latest hidden size
        self.head = nn.Linear(in_features, num_classes)
        # init weights
        
        self.adapter.apply(init_weights)
        self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)
        
    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []

        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_features.append(
                per_adapter(per_output[per_model.prefix][FEATURES])
            )
            
        features = self.fusion_mlp(torch.cat(multimodal_features, dim=1))
        logits = self.head(features)
        
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }
        
        return fusion_output


def create_model(config, num_classes):
    models = []
    for model_name in config.names:
        model_config = getattr(config, model_name)
        if model_name.lower().startswith(TIMM_MODEL):
            model = TIMM(
                    prefix = model_name,
                    model_name = model_config.model_name,
                    num_classes = num_classes,
                    pretrained = model_config.pretrained if hasattr(model_config, "pretrained") else True,
                    prenorm = model_config.prenorm if hasattr(model_config, "prenorm") else False,
            )
        elif model_name.lower().startswith('mem_vit'):
            model = MEMORYVIT(
                    prefix = model_name,
                    num_classes = num_classes,
            )
        elif model_name.lower().startswith(FUSION_MLP):
            fusion_model = functools.partial(
                    FusionMLP,
                    prefix = model_name,
                    num_classes = num_classes,
                    hidden_features = model_config.hidden_features,
                    adapt_in_features = model_config.adapt_in_features,
                    activation = model_config.activation,
                    dropout_prob = model_config.dropout_prob,
                    normalization = model_config.normalization,
            )
        else:
            raise ValueError(f"unknown model name: {model_name}")
        
        
        
        models.append(model)
        
    if len(models) == 1:
        return models[0]
    else:
        return fusion_model(models=models)