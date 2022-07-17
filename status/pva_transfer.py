from .state import ELExceptionRaise
from .state_transfer import StateTrasferFunc
import numpy as np
class PVStateTransferFunc(StateTrasferFunc):
    def __init__(self):
        A=np.array([[0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
        B=np.array([[0,0,0],
                     [0,0,0],
                     [0,0,0],
                     [1,0,0],
                     [0,1,0],
                     [0,0,1]])
        super(PVStateTransferFunc,self).__init__(["positionx","positiony","positionz","velocityx","velocityy","velocityz"],
                                                 ["acceleratex","acceleratey","acceleratez"],A,B)



    def _mapStatusandInput(self,stateDict,input):
        try:
            ELExceptionRaise(self.stateName,stateDict,"State")
            ELExceptionRaise(self.inputName,input,"Input")
        except Exception as e:
            raise e

        state=np.ndarray([len(self.stateName),1])
        inp= np.ndarray([len(self.inputName),1])

        for index,k in enumerate(self.stateName):
            state[index]=stateDict[k]

        for index,k in enumerate(self.inputName):
            inp[index]=input[k]


        return state,inp
