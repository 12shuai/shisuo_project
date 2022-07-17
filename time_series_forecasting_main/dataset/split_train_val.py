import os
import random
import shutil
from tkinter import mainloop
from tkinter.tix import MAIN

from pip import main

def split_train_val_dataset(root_path,save_path,split_ratio):
    class_list=os.listdir(root_path)
    for cls in class_list:
        train_path=os.path.join(save_path,"train",cls)
        val_path=os.path.join(save_path,"val",cls)
        create_dir([train_path,val_path])
        train_file_list,val_file_list=get_train_file_stem(os.path.join(root_path,cls,"csv"),split_ratio)
        split_dataset(root_path,save_path,cls,train_file_list,val_file_list)



def create_dir(paths):
    for path in paths:
        os.makedirs(os.path.join(path,"csv"),exist_ok=True)
        os.makedirs(os.path.join(path,"image"),exist_ok=True)
    

def get_train_file_stem(path,split_ratio):
    file_name_list=os.listdir(path)
    random.shuffle(file_name_list)
    train_length=int(len(file_name_list)*split_ratio)


    train_name_list=file_name_list[:train_length]
    val_name_list=file_name_list[train_length:]
    return [name.split(".")[0] for name in train_name_list],[name.split(".")[0] for name in val_name_list]



def split_dataset(root_path,save_path,cls,train_file_list,val_file_list):
    src_path=os.path.join(root_path,cls)
    train_path=os.path.join(save_path,"train",cls)
    val_path=os.path.join(save_path,"val",cls)

    for train_file in train_file_list:
        train_csv=os.path.join(src_path,"csv",train_file+".csv")
        train_image=os.path.join(src_path,"image",train_file+".jpg")
        shutil.copy(train_csv,os.path.join(train_path,"csv",train_file+".csv"))
        try:
            shutil.copy(train_image,os.path.join(train_path,"image",train_file+".jpg"))
        except:
            print(os.path.join(train_path,"image",train_file+".jpg"),"not exists")

    for val_file in val_file_list:
        val_csv=os.path.join(src_path,"csv",val_file+".csv")
        val_image=os.path.join(src_path,"image",val_file+".jpg")
        shutil.copy(val_csv,os.path.join(val_path,"csv",val_file+".csv"))
        try:
            shutil.copy(val_image,os.path.join(val_path,"image",val_file+".jpg"))
        except:
            print(os.path.join(train_path,"image",train_file+".jpg"),"not exists")




if __name__=="__main__":
    seed=43
    random.seed(seed)
    split_ratio=0.7
    root_path="output/output-200-2nd"
    save_path="train_val_split"
    split_train_val_dataset(root_path,save_path,split_ratio)