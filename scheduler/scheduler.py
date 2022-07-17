from simulater import Simulater, Simulater2, Simulater4
from .environment import Environment

class Scheduler:
    """继承randomInitState"""
    def __init__(self,condition,setting,lenth,stepperFunc,interval):
        self.stepperFunc=stepperFunc
        self.condition=condition
        self.environment=Environment(setting,lenth)
        self.setting = setting
        self.interval=interval

    def forward(self):
        for initState,inputMaker,handler in self.environment:
            
            simulater=Simulater(initState,self.condition,self.stepperFunc,self.interval,inputMaker,handler)
            simulater.forward()

    def forward2(self):
        for initState,inputMaker1,inputMaker2,handler in self.environment:
            
            simulater=Simulater2(initState,self.condition,self.stepperFunc,self.interval,inputMaker1,inputMaker2,handler)
            simulater.forward()

    def forward4(self):
        for initState,inputMaker1,inputMaker2,inputMaker3,inputMaker4,handler in self.environment:
            
            simulater=Simulater4(initState,self.condition,self.stepperFunc,self.interval,inputMaker1,inputMaker2,inputMaker3,inputMaker4,handler)
            simulater.forward()

   
