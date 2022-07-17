import os




# PATH=os.path.join("train_val_split","train")
PATH=os.path.join("train_val_split","val")
if __name__=="__main__":
    dirnames=os.listdir(PATH)
    dirpaths=[os.path.join(PATH,name) for name in dirnames]

    total=0
    for dirpath in dirpaths:
        count=len(os.listdir(os.path.join(dirpath,"csv")))
        print(os.path.split(dirpath)[1],":",count)
        total+=count
    
    print("total:",total)
    