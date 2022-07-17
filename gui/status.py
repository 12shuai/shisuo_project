class Status:
    def __init__(self):
        self.name=self.__class__

    def next(self):
        raise NotImplementedError()
    
    
    

class NotChooseFileStatus(Status):
    def __init__(self):
        super().__init__()

    def next(self,right):
        if right:


class 
