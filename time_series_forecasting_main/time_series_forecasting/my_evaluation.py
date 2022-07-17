import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
import os
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose
import my_model
from torch.nn import CrossEntropyLoss

class NormalizeTransform:
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
    
    def _getClassesName(self):
        self.classesName=os.listdir(self.root_path)


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



def evaluate(
    root_path: str,
    transforms: list,
    pretrained_path: str,
    batch_size:int,
    class_name:list,
):
    """
    Evaluates the model on the last 8 labeled weeks of the data.
    Compares the model to a simple baseline : prediction the last known value
    :param data_csv_path:
    :param feature_target_names_path:
    :param trained_json_path:
    :param eval_json_path:
    :param horizon_size:
    :param data_for_visualization_path:
    :return:
    """
    val_data = BasicEventDataset(
        os.path.join(root_path,"val"),transforms,class_name
    )

    print("len(val_data)", len(val_data))
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    model = my_model.TimeSeriesForcasting(
        n_encoder_inputs=6,
        classes_num=len(val_data.classesName),
        lr=1e-5,
        dropout=0.1
    )

    device="cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(pretrained_path,map_location=device)["state_dict"])
    model.to(device)
    model.eval()

    total_loss={}
    total_recall={}
    total_presision={}
    for cls in val_data.classesName:
        total_loss[cls]=[0,0]
        total_recall[cls]=[0,0]
        total_presision[cls]=[0,0]

    with torch.no_grad():
        for i, (src,label) in tqdm(enumerate(val_loader)):
            src=src.to(model.device)
            label=label.to(model.device)

            prediction = model(src)
            _,predict_cls_index=torch.max(prediction,dim=1)

            loss=CrossEntropyLoss()(prediction,label)
            predict_ok=1 if predict_cls_index==label else 0
            
            loss_item=loss.cpu().item()
            label_index=label.cpu().item()
  
            total_loss[val_data.classesName[label_index]][0]+=loss_item
            total_loss[val_data.classesName[label_index]][1]+=1
            
            total_recall[val_data.classesName[label_index]][0]+=predict_ok
            total_recall[val_data.classesName[label_index]][1]+=1

            total_presision[val_data.classesName[predict_cls_index.cpu().item()]][0]+=predict_ok
            total_presision[val_data.classesName[predict_cls_index.cpu().item()]][1]+=1

            if i>100:
                break
        
        
    
    for key in total_loss.keys():
        loss_value,loss_num=total_loss[key]
        total_loss[key]=loss_value/loss_num if loss_num>0 else 0
        

        accuracy_value,accuracy_num=total_recall[key]
        total_recall[key]=accuracy_value/accuracy_num if accuracy_num>0 else 0
        
        
        presision_value,presision_num=total_presision[key]
        total_presision[key]=presision_value/presision_num if presision_num>0 else 0

    total_loss["total"]=sum(total_loss.values())/len(total_loss)
    total_recall["total"]=sum(total_recall.values())/len(total_recall)
    total_presision["total"]=sum(total_presision.values())/len(total_presision)
        
    
    return total_loss,total_recall,total_presision

        
        

if __name__ == "__main__":
    import argparse

    os.chdir(os.path.join(os.getcwd(), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",default="train_val_split")##"output/output-200-2nd"
    parser.add_argument("--transforms",type=list,default=["NormalizeTransform()"])
    parser.add_argument("--batch_size",default=1)
    parser.add_argument("--pre_trained",type=str,default="models/ts_views_models/transformer_split.ckpt")##"models/ts_views_models/not_split.ckpt"
    args = parser.parse_args()

    total_loss,total_recall,total_presision=evaluate(
        root_path=args.root_path,
        transforms=args.transforms,
        pretrained_path=args.pre_trained,
        batch_size=args.batch_size,
        class_name=['SOMERSAULT', 'DOWN_RIGHT', 'SPIRAL', 'SHARPTURN_RIGHT', 'LEFT', 'DOWN_LEFT', 'DOWN', 'UP_RIGHT', 'UP',
         'STRAIGHT', 'HALF_SOMERSAULT', 'SERPENTINE', 'UP_LEFT', 'RIGHT', 'SHARPTURN_LEFT']
    )

    print(total_recall,"\n",total_presision)