IMGTRAINER targets for using an ensemble of TIMM models for binary classification.

- img_model.py : create the ensemble models using TIMM backbones, currently we support MLP and Transformer fusion layers.
- img_set.py : create the datasets.
- img_eval.py : evaluation codes.
- img_train.py : train the models.

To train the model, please run `main.py`. 
For tuning the hyper-parameters, use the `config.yaml`:
    - data.train_root: path to the training set.
    - data.test_root: path to the test set.
    - models.names: the model names (a list) used for ensembling.
    - model.{model_name}.model_name: should be a model supported by timm e.g., resnet18, res2next50. Use ImageNetPretrained model when `pretrained=True`.
    - opt.scheduler: support cosine, epoential and const. 


