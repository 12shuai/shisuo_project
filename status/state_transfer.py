import numpy as np
from .state import StatusDict,Status
from .state import ELExceptionRaise
class Stepper:
    MAXI_INTERVAL=1E-1
    MINI_INTERVAL=1E-8
    def __init__(self,func,interval):
        if not isinstance(func,StateTrasferFunc):
            raise TypeError(f"{func}(type{type(func)}) should be StateTranferFunc type")
        if not self.MINI_INTERVAL<interval<self.MAXI_INTERVAL:
            raise Exception(f"The interval should in [{self.MINI_INTERVAL},{self.MAXI_INTERVAL}]")
        self.func=func
        self.interval=interval

    def setInterval(self,interval):
        self.interval=interval

    def __call__(self,status,input):
        update=self.func(status,input)*self.interval

        return status+update

class StateTrasferFunc:
    """需要重写 _mapStatusandInput(self,stateDict,input)，并定义stateName,InputName,（有序的字典对象，对应A,B转移矩阵）A，B转移矩阵"""
    def __init__(self,stateName,inputName,A,B):

        if not isinstance(stateName,list) or not  isinstance(inputName,list):
            raise TypeError("Both inputs should be dict(python) or list type")
        self.stateName=stateName
        self.inputName=inputName
        self.A=A
        self.B=B

    def check(self,stateDict,input):
        """判断输入的状态字典的键是否正确，错误应该raise 错误原因"""
        try:
            ELExceptionRaise(self.stateName, stateDict, "State")
            ELExceptionRaise(self.inputName, input, "Input")
        except Exception as e:
            raise e


    def __call__(self, state,input):
        try:
            self.check(state,input)
        except Exception as e:
            raise e
        return self.forward(state,input)

    def forward(self,state,input):
        """根据当前状态和输入，输出更新值"""
        state,input=self._mapStatusandInput(state,input)
        resNp=self._forward(state,input)
        res=StatusDict()
        for index,k in enumerate(self.stateName):
            res.append(Status(k,resNp[index][0]))

        return res


    def printStateName(self):
        for index,k,v in enumerate(self.stateName.items()):
            print(f"The {index}th state:{k}")


    def printInputName(self):
        for index,k,v in enumerate(self.inputName.items()):
            print(f"The {index}th input:{k}")




    def _forward(self,state,input):

        update=np.matmul(self.A,state)+np.matmul(self.B,input)

        return update


    def _mapStatusandInput(self,stateDict,input):
        """根据当前状态和输入，输出更新值"""
        raise NotImplementedError()







