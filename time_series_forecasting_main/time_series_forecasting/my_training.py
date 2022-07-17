import json
import random
import os
from os import environ
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python" 
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose
from . import my_model
from os import environ
import torch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"


class NormalizeTransform(torch.nn.Module):
    def __init__(self,eplsion=1e-8):
        self.eplsion=eplsion
        pass

    def __call__(self,status):

        status=(status-torch.mean(status,dim=0,keepdim=True))/(torch.std(status,dim=0,keepdim=True)+self.eplsion)

        return status

class BasicEventDataset(Dataset):
    def __init__(self,root_path,transforms,class_name):
        super(BasicEventDataset,self).__init__()
        self.root_path=root_path
        self.transform=None
        self.classesName=None
        self.getTransforms(transforms)
        self.dataPathList=[]

        self.classesName=class_name

        self._getDataPathList()

    def getTransforms(self,transforms):
        self.transform=self._getTransforms(transforms)


    def _getDataPathList(self):
        for cls in self.classesName:
            path=os.path.join(self.root_path,cls)
            csvList=os.listdir(os.path.join(path,'csv'))
            self.dataPathList.extend([os.path.join(path,'csv',csvName) for csvName in csvList])
        




    def _getTransforms(self,transforms):
        if transforms is None:
            return None
        if isinstance(transforms,str):
            return eval(transforms)
        elif isinstance(transforms,list):
            transforms=[eval(i) for i in transforms]
            return Compose(transforms)
        elif isinstance(transforms,torch.nn.Module):
            return transforms
        else:
            raise NotImplementedError("The transforms should be str or list")


    def __len__(self):
        return len(self.dataPathList)


    def __getitem__(self, item):
        path=self.dataPathList[item]
        label=self.classesName.index(path.split(os.path.sep)[-3])
        data=np.array(pd.read_csv(path))
        data= torch.tensor(data, dtype=torch.float)
        if self.transform:
            data=self.transform(data)
        label= torch.tensor(label, dtype=torch.long)
        
        return np.array(data),label


    # def collate_fn(self,batch_data):

    #     inputs, labels = list(zip(*batch_data))
    #     sorted(inputs, key=lambda xi: xi.shape[0], reverse=True)
    #     sent_seq = [xi for xi in inputs]

    #     padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
    #     return padded_sent_seq, torch.stack(labels)


def train(
    model_name,
    root_path: str,
    transforms:list,
    output_json_path: str,
    pretrained_path:str,
    class_name:list,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    batch_size: int = 32,
    epochs: int = 2000,
    
):
    ##1.准备训练集测试集
    train_data = BasicEventDataset(
        os.path.join(root_path,"train"),transforms,class_name
    )
    val_data = BasicEventDataset(
        os.path.join(root_path,"val"),transforms,class_name
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
    )


    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )

    ##2.准备模型，以及是否加载预训练模型
    if model_name=="transformer":
        model = my_model.TimeSeriesForcasting(
            n_encoder_inputs=6,
            classes_num=len(train_data.classesName),
            lr=1e-5,
            dropout=0.1
        )
    elif model_name=="lstm":
        model = my_model.LSTMTimeSeriesForcasting(
            input_size=6,hidden_size=16,num_classes=len(train_data.classesName),num_layers=5,strategy="last"
        )

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path)["state_dict"])
    
    logger = TensorBoardLogger(
        save_dir=log_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )

    ##3.准备训练器
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    ##4.开始训练
    trainer.fit(model, train_loader, val_loader)

    ##5.开始测试
    result_val = trainer.test(test_dataloaders=val_loader)
    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json



if __name__ == "__main__":
    import argparse

    ##1.训练参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",default="train_val_split")##"output/output-200-2nd"
    parser.add_argument("--transforms",type=list,default=["NormalizeTransform()"])
    parser.add_argument("--output_json_path", default=None)
    parser.add_argument("--log_dir",default="models/ts_views_logs")
    parser.add_argument("--model_dir",default="models/ts_views_models")
    parser.add_argument("--batch_size",default=1)
    parser.add_argument("--seed",default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model",type=str,default="transformer")#
    parser.add_argument("--pre_trained",type=str,default="models/ts_views_models/transformer_split.ckpt")##"models/ts_views_models/not_split.ckpt"
    args = parser.parse_args()

    ##2.设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ##3.开始训练
    output=train(
        model_name=args.model,
        root_path=args.root_path,
        transforms=args.transforms,
        output_json_path=args.output_json_path,
        pretrained_path=args.pre_trained,
        class_name=['SOMERSAULT', 'DOWN_RIGHT', 'SPIRAL', 'SHARPTURN_RIGHT', 'LEFT', 'DOWN_LEFT', 'DOWN', 'UP_RIGHT', 'UP', 'STRAIGHT', 'HALF_SOMERSAULT', 'SERPENTINE', 'UP_LEFT', 'RIGHT', 'SHARPTURN_LEFT'],
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    ##4.打印输出
    print(output)