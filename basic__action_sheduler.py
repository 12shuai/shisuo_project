"""
    >>> initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":100})
    >>> interval=0.0005
    >>> lenth=100
    >>> handlers=[PositionHandler(),CSVHandler("a.csv")]
    >>> uniformScheduler=UniformScheduler(initState,interval,lenth,handlers)
    >>> uniformScheduler.forward()
"""

from status import PVStateTransferFunc,StatusDict
from simulater import Simulater
from inputmaker import *
from  handler import PositionHandler,CSVHandler

from utils import *


class BasicScheduler(Simulater):

    def __init__(self,initStatus,interval,inputMaker,handler=None):
        self.name=self.__class__
        super(BasicScheduler,self).__init__(initStatus,PVStateTransferFunc(),interval,inputMaker,handler)


    # def forward(self):
    #     state=self.initStatus
    #     for input in self.inputMaker:
    #         state=self.stepper(state,input)
    #         self.recorder.append(state)
    #
    #     self.handle()

##1.匀速直线
class UniformScheduler(BasicScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":100})
        >>>interval=0.0005
        >>>lenth=100
        >>>handlers=[PositionHandler(),CSVHandler("a.csv")]
        >>>uniformScheduler=UniformScheduler(initState,interval,lenth,handlers)
        >>>uniformScheduler.forward()
    """
    def __init__(self,initStatus,interval,lenth,handler=None):
        inputMaker=ConstantAccelerateMaker(lenth=lenth)
        super(UniformScheduler,self).__init__(initStatus,interval,inputMaker,handler)




##1.2 匀速前飞
class ForwardUniformScheduler(UniformScheduler):
    """
        >>>initState = StatusDict({"positionx": 500, "positiony": 400, "positionz": 0})
        >>>initVelocity={ "velocityx": 100, "velocityy": 100, "velocityz": 100}
        >>>interval = 0.0005
        >>>lenth = 100
        >>>handlers = [CSVHandler("a.csv")]
        >>>forwardUniformScheduler = ForwardUniformScheduler(initState, initVelocity,interval, lenth, handlers)
        >>>forwardUniformScheduler.forward()
    """
    def __init__(self,initPosition,initVelocity,interval,lenth,handler=None):
        initStatus=initPosition.copy()
        initStatus.update(initVelocity)
        initStatus["velocityz"]=0
        super(ForwardUniformScheduler,self).__init__(initStatus,interval,lenth,handler)

##1.3 随机匀速直飞(需要给定幅值)
class RandomForwardUniformScheduler(UniformScheduler):
    """
        >>>initState = StatusDict({"positionx": 500, "positiony": 400, "positionz": 0})
        >>>interval = 0.0005
        >>>lenth = 100
        >>>handlers = [PositionHandler()]
        >>>forwardUniformScheduler = RandomForwardUniformScheduler(initState, 500, interval, lenth, handlers)
        >>>forwardUniformScheduler.forward()
    """
    def __init__(self,initPosition,initVelocity,interval,lenth,handler=None):
        initStatus=initPosition.copy()
        initStatus["velocityx"]=0
        initStatus.randomKey("velocityx",min=-initVelocity,max=initVelocity)
        initStatus["velocityy"]=math.sqrt(initVelocity**2-initStatus["velocityx"]**2)
        initStatus["velocityz"]=0
        super(RandomForwardUniformScheduler,self).__init__(initStatus,interval,lenth,handler)



##1.4 随机匀速上飞(需要给定幅值)
class RandomUpUniformScheduler(UniformScheduler):
    """
        >>>initState = StatusDict({"positionx": 500, "positiony": 400, "positionz": 0})
        >>>interval = 0.0005
        >>>lenth = 100
        >>>handlers = [PositionHandler()]
        >>>upUniformScheduler = RandomUpUniformScheduler(initState, 500, interval, lenth, handlers)
        >>>upforwardUniformScheduler.forward()
    """
    def __init__(self,initPosition,initVelocity,interval,lenth,handler=None):
        initStatus=initPosition.copy()

        initStatus["velocityy"]=0
        initStatus["velocityz"]=0
        initStatus.randomKey("velocityz",min=0,max=initVelocity/10)
        remain=math.sqrt(initVelocity**2-initStatus["velocityz"]**2)
        initStatus.randomKey("velocityy",min=-remain,max=remain)
        initStatus["velocityx"] = math.sqrt(remain**2-initStatus["velocityy"]**2)

        super(RandomUpUniformScheduler,self).__init__(initStatus,interval,lenth,handler)

##1.5 随机匀速俯冲(需要给定幅值)
class RandomDownUniformScheduler(UniformScheduler):
    """
        >>>initState = StatusDict({"positionx": 500, "positiony": 400, "positionz": 0})
        >>>interval = 0.0005
        >>>lenth = 100
        >>>handlers = [PositionHandler()]
        >>>downUniformScheduler = RandomDownUniformScheduler(initState, 500, interval, lenth, handlers)
        >>>downforwardUniformScheduler.forward()
    """
    def __init__(self,initPosition,initVelocity,interval,lenth,handler=None):
        initStatus=initPosition.copy()

        initStatus["velocityy"]=0
        initStatus["velocityz"]=0
        initStatus.randomKey("velocityz",min=-initVelocity/10,max=0)
        remain=math.sqrt(initVelocity**2-initStatus["velocityz"]**2)
        initStatus.randomKey("velocityy",min=-remain,max=remain)
        initStatus["velocityx"] = math.sqrt(remain**2-initStatus["velocityy"]**2)

        super(RandomDownUniformScheduler,self).__init__(initStatus,interval,lenth,handler)


#2.变加速





#3.转弯
##3.1 匀加速转弯
class TurnScheduler(BasicScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":100})
        >>>interval=0.02
        >>>lenth=100
        >>>handlers=[PositionHandler()]
        >>>initAcc={"acceleratex":-40,"acceleratey":-10,"acceleratez":60}
        >>>turnScheduler=TurnScheduler(initState,initAcc,interval,lenth,handlers)
        >>>turnScheduler.forward()
    """
    def __init__(self,initStatus,initAcclerate,interval,lenth,handler=None):
        inputMaker=ConstantAccelerateMaker(initState=initAcclerate,lenth=lenth)
        super(TurnScheduler,self).__init__(initStatus,interval,inputMaker,handler)



##3.2 随机匀加速转弯(需要给定幅值)
class RandomTurnScheduler(TurnScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":0})
        >>>interval=0.02
        >>>lenth=100
        >>>handlers=[PositionHandler()]
        >>>initAcc=80
        >>>randomTurnScheduler=RandomTurnScheduler(initState,initAcc,interval,lenth,handlers)
        >>>randomTurnScheduler.forward()
    """
    def __init__(self,initStatus,initAcc,interval,lenth,handler=None):
        initAcclerate=StatusDict({"acceleratex":0,"acceleratey":0,"acceleratez":0})
        initAcclerate.randomKey("acceleratex",min=-initAcc,max=initAcc)
        remain=math.sqrt(initAcc**2-initAcclerate["acceleratex"]**2)
        initAcclerate.randomKey("acceleratey",min=-remain,max=remain)
        initAcclerate["acceleratez"] = math.sqrt(remain**2-initAcclerate["acceleratey"]**2)
        super(RandomTurnScheduler,self).__init__(initStatus,initAcclerate,interval,lenth,handler)


##3.3 随机左匀加速转弯(需要给定幅值)
class RandomLeftTurnScheduler(TurnScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":0})
        >>>interval=0.02
        >>>lenth=100
        >>>handlers=[PositionHandler()]
        >>>initAcc=150
        >>>randomLeftTurnScheduler=RandomLeftTurnScheduler(initState,initAcc,interval,lenth,handlers)
        >>>randomLeftTurnScheduler.forward()
    """
    def __init__(self,initStatus,initAcc,interval,lenth,handler=None):
        initAcclerate=StatusDict({"acceleratex":0,"acceleratey":0,"acceleratez":0})

        angle=math.atan2(initStatus["velocityy"],initStatus["velocityx"])
        angle=randomLeft(angle)
        signX=getXSignByAngle(angle)
        tan=math.tan(angle)
        initAcclerate["acceleratex"]=abs(math.sqrt(initAcc**2/(1+tan**2)))*signX
        initAcclerate["acceleratey"] =initAcclerate["acceleratex"]*tan
        print(initAcclerate)
        super(RandomLeftTurnScheduler,self).__init__(initStatus,initAcclerate,interval,lenth,handler)


##3.4 随机右匀加速转弯(需要给定幅值)
class RandomRightTurnScheduler(TurnScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":0})
        >>>interval=0.02
        >>>lenth=100
        >>>handlers=[PositionHandler()]
        >>>initAcc=150
        >>>randomRightTurnScheduler=RandomRightTurnScheduler(initState,initAcc,interval,lenth,handlers)
        >>>randomRightTurnScheduler.forward()
    """
    def __init__(self,initStatus,initAcc,interval,lenth,handler=None):
        initAcclerate = StatusDict({"acceleratex": 0, "acceleratey": 0, "acceleratez": 0})
        angle = math.atan2(initStatus["velocityy"], initStatus["velocityx"])
        angle = randomRight(angle)
        signX = getXSignByAngle(angle)
        tan = math.tan(angle)
        initAcclerate["acceleratex"] = abs(math.sqrt(initAcc**2 / (1 + tan ** 2))) * signX
        initAcclerate["acceleratey"] = initAcclerate["acceleratex"] * tan
        super(RandomRightTurnScheduler, self).__init__(initStatus, initAcclerate, interval, lenth, handler)


##3.4 随机向上匀加速转弯(需要给定幅值)
class RandomUpTurnScheduler(TurnScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":0})
        >>>interval=0.02
        >>>lenth=100
        >>>handlers=[PositionHandler()]
        >>>initAcc=150
        >>>randomUpTurnScheduler=RandomUpTurnScheduler(initState,initAcc,interval,lenth,handlers)
        >>>randomUpTurnScheduler.forward()
    """
    def __init__(self,initStatus,initAcc,interval,lenth,handler=None):
        initAcclerate = StatusDict({"acceleratex": 0, "acceleratey": 0, "acceleratez": 0})
        initAcclerate["acceleratez"]=randomMinMax(0,initAcc)
        remain=math.sqrt(initAcc**2-initAcclerate["acceleratez"]**2)
        sign=getSign(initStatus["velocityx"])
        tan = initStatus["velocityy"]/initStatus["velocityx"]
        initAcclerate["acceleratex"]=math.sqrt(remain**2/(1+tan**2))*sign
        initAcclerate["acceleratey"] = initAcclerate["acceleratex"] * tan
        super(RandomUpTurnScheduler, self).__init__(initStatus, initAcclerate, interval, lenth, handler)


##3.5 随机向下匀加速转弯(需要给定幅值)
class RandomDownTurnScheduler(TurnScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":0})
        >>>interval=0.02
        >>>lenth=100
        >>>handlers=[PositionHandler()]
        >>>initAcc=150
        >>>randomDownTurnScheduler=RandomDownTurnScheduler(initState,initAcc,interval,lenth,handlers)
        >>>randomDownTurnScheduler.forward()
    """
    def __init__(self,initStatus,initAcc,interval,lenth,handler=None):
        initAcclerate = StatusDict({"acceleratex": 0, "acceleratey": 0, "acceleratez": 0})
        initAcclerate["acceleratez"]=randomMinMax(-initAcc,0)
        remain=math.sqrt(initAcc**2-initAcclerate["acceleratez"]**2)
        sign=getSign(initStatus["velocityx"])
        tan = initStatus["velocityy"]/initStatus["velocityx"]
        initAcclerate["acceleratex"]=math.sqrt(remain**2/(1+tan**2))*sign
        initAcclerate["acceleratey"] = initAcclerate["acceleratex"] * tan
        super(RandomDownTurnScheduler, self).__init__(initStatus, initAcclerate, interval, lenth, handler)



##3.4 随机向上匀加速转弯(需要给定幅值)
class RandomLeftUpTurnScheduler(TurnScheduler):
    """
        >>>initState=StatusDict({"positionx":500,"positiony":400,"positionz":0,"velocityx":100,"velocityy":100,"velocityz":0})
        >>>interval=0.02
        >>>lenth=100
        >>>handlers=[PositionHandler()]
        >>>initAcc=150
        >>>randomUpTurnScheduler=RandomUpTurnScheduler(initState,initAcc,interval,lenth,handlers)
        >>>randomUpTurnScheduler.forward()
    """
    def __init__(self,initStatus,initAcc,interval,lenth,handler=None):
        initAcclerate = StatusDict({"acceleratex": 0, "acceleratey": 0, "acceleratez": 0})
        initAcclerate["acceleratez"]=randomMinMax(0,initAcc)
        remain=math.sqrt(initAcc**2-initAcclerate["acceleratez"]**2)
        sign=getSign(initStatus["velocityx"])
        tan = initStatus["velocityy"]/initStatus["velocityx"]
        initAcclerate["acceleratex"]=math.sqrt(remain**2/(1+tan**2))*sign
        initAcclerate["acceleratey"] = initAcclerate["acceleratex"] * tan
        super(RandomUpTurnScheduler, self).__init__(initStatus, initAcclerate, interval, lenth, handler)


if __name__ == '__main__':
    # import doctest
    # doctest.testmod(verbose=True)

    initState = StatusDict(
        {"positionx": 500, "positiony": 400, "positionz": 0, "velocityx": 100, "velocityy": 100, "velocityz": 0})
    interval = 0.02
    lenth = 100
    handlers = [PositionHandler()]
    initAcc = 150
    randomRightTurnScheduler = RandomRightTurnScheduler(initState, initAcc, interval, lenth, handlers)
    randomRightTurnScheduler.forward()