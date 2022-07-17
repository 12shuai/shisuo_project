import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torchvision.transforms import *






class NormalizeTransform:
    def __init__(self,eplsion=1e-8):
        self.eplsion=eplsion
        pass

    def __call__(self,status):

        status=(status-torch.mean(status,dim=0,keepdim=True))/(torch.std(status,dim=0,keepdim=True)+self.eplsion)

        return status




class BasicEventDataset(Dataset):
    def __init__(self,root_path,transforms):
        super(BasicEventDataset,self).__init__()
        self.root_path=root_path
        self.transform=None
        self.classesName=None
        self.getTransforms(transforms)
        self.dataPathList=[]

        self._getClassesName()

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
        data=pd.read_csv(path)
        data= torch.tensor(data, dtype=torch.float)
        label= torch.tensor(label, dtype=torch.long)
        return np.array(data),label


    def collate_fn(self,batch_data):

        inputs, labels = list(zip(*batch_data))
        sorted(inputs, key=lambda xi: xi.shape[0], reverse=True)
        sent_seq = [xi for xi in inputs]

        padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
        return padded_sent_seq, torch.stack(labels)










if __name__ == '__main__':
    from configs.defaults import _C
    C=_C.clone()
    C.TRAIN.DATALOADER.DATASET.PARAMETERS.ROOT_PATH = "../actiondata/Lag pursuit_0_action"
    a=ActionDataset(C.TRAIN.DATALOADER.DATASET.PARAMETERS)
    # dataloader=
    # for i in range(100):
    #     pass







