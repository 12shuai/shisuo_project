from status import dict2statedict,StatusDict
from utils import *
import math
from .inputmaker import InputMaker

##直线类别
HS=0
US=1
DS=2

AS=3
NAS=4

HAS=5
UAS=6
DAS=7

HNAS=8
UNAS=9
DNAS=10

##转弯类别
L = 0
U = 1
R = 2
D = 3
UL = 4
DL = 5
UR = 6
DR = 7





##0.加速模型
class AccelerateMaker(InputMaker):
    def __init__(self,initState,lenth=None,tracer=None):
        super(AccelerateMaker,self).__init__({"acceleratex","acceleratey","acceleratez"},initState,lenth,tracer)
    def _produce(self):
        raise NotImplementedError()


##1.无/常加速度模型
class ConstantAccelerateMaker(AccelerateMaker):
    def __init__(self,initState=None,lenth=None,tracer=None):
        if not initState:
            initState={"acceleratex":0,"acceleratey":0,"acceleratez":0}
        super(ConstantAccelerateMaker,self).__init__(initState,lenth,tracer)
    def _produce(self):
        return self.initState


##2.变加速度模型
class VarAccelerateMaker(AccelerateMaker):

    def __init__(self,initState=None,lenth=None,tracer=None):
        """scheduler接受StateDict，或者dict为输入，并输出下一时刻的加速度"""
        if not initState:
            initState=dict2statedict({"acceleratex":0,"acceleratey":0,"acceleratez":0})

        def decorate(f,state):
            init=True
            def ff():
                nonlocal state,init
                if init:
                    init=False
                    return state
                state=f(state)
                return state

            return ff
        try:
            self.scheduler=decorate(self._scheduler,initState)
        except AttributeError:
            raise Exception("You must implement the _scheduler(self,state) function")
        except Exception as e:
            raise e
        super(VarAccelerateMaker,self).__init__(initState,lenth,tracer)


    def _produce(self):

        return self.scheduler()


##2.1指定方向直线运动模型
class StraightAccelerateMaker(VarAccelerateMaker):
    def __init__(self,orient,max,float=0.1,initState=None,lenth=None,tracer=None):
        """scheduler接受StateDict，或者dict为输入，并输出下一时刻的加速度"""
        self._checkOrient(orient)
        self.orient=orient
        self.max = max
        self.float=float
        super(StraightAccelerateMaker,self).__init__(initState,lenth,tracer)

    def _scheduler(self, state):
        initStatus = self.tracer[-1]
        velocity=initStatus.getSubStatus(["velocityx","velocityy","velocityz"])
        # vz=velocity["velocityz"]
        # if self.orient in [US,UAS,UNAS]:

        if state.norm()<=0.01:
            state=StatusDict({"acceleratex":velocity["velocityx"],
                             "acceleratey":velocity["velocityy"],
                            "acceleratez":velocity["velocityz"]})
            
            state=state.normVector()*math.fabs(self.max*math.sqrt(3)*self.float)

        theta=velocity.getTheta(state)
        updateTheta=math.fabs(math.pi*self.float)
        theta2=max(0,theta-updateTheta)
        updateTheta=theta-theta2
        nextState=rotate2Vec(state,velocity,updateTheta)


        oldAXYNorm = state.norm()
        updateA=math.fabs(self.max*math.sqrt(3)*self.float)
        newANorm = randomMinMax(max(0, oldAXYNorm - updateA),
                                  min(oldAXYNorm + updateA, self.max * math.sqrt(3)))


        return nextState.normVector()*newANorm
    def _checkOrient(self,orient):
        if not 0<=orient<=10:
            raise Exception("The orient should be in [0,10]\n"+self.orientString())


    def orientString(self):
        return "0:HS" \
               "1:US" \
               "2:DS" \
               "3:AS"\
               "4:NAS"\
               "5:HAS" \
               "6:UAS" \
               "7:DAS" \
               "8:HNAS" \
               "9:UNAS" \
               "10:DNAS" \
            ##需要随机初始化

##2.2规定方向的拐弯模型
class OrientTurnAccelerateMaker(VarAccelerateMaker):
    def __init__(self,orient,max,float=0.1,initState=None,lenth=None,tracer=None):
        """scheduler接受StateDict，或者dict为输入，并输出下一时刻的加速度"""
        self._checkOrient(orient)
        self.orient=orient
        self.max = max
        self.float=float
        super(OrientTurnAccelerateMaker,self).__init__(initState,lenth,tracer)

    def _scheduler(self,state):
        initStatus=self.tracer[-1]
        nextState=state.zero()
        ##1.确定acceleratez

        oldAZ = state["acceleratez"]
        updateAZ=math.fabs(self.float*self.max)
        if self.orient in [U,UL,UR]:
            if -self.max<=oldAZ<=0:
                nextState["acceleratez"]=oldAZ+updateAZ
            elif 0<oldAZ<=self.max:
                nextState["acceleratez"] = randomMinMax(max(0,oldAZ-updateAZ), min(self.max,oldAZ+updateAZ))

        elif self.orient in [D,DL,DR]:
            if 0<= oldAZ <= self.max :
                nextState["acceleratez"] = oldAZ -updateAZ
            elif -self.max <=oldAZ <0:
                nextState["acceleratez"] = randomMinMax(max(-self.max, oldAZ-updateAZ),
                                                        min(0, oldAZ+updateAZ))
        else:

            if initStatus["velocityz"]>0:
                nextState["acceleratez"] = - updateAZ

            else:
                nextState["acceleratez"] = updateAZ



        if oldAZ < -self.max:
            nextState["acceleratez"] = -self.max
        elif oldAZ>self.max:
            nextState["acceleratez"] = self.max

        ##2.确定acceleratex和acceleratey
        vAngle = math.atan2(initStatus["velocityy"], initStatus["velocityx"])
        # print("velocity:",initStatus["velocityx"], initStatus["velocityy"],initStatus["velocityz"])

        aAngle = math.atan2(state["acceleratey"], state["acceleratex"])
        # print("acc",state["acceleratex"], state["acceleratey"], state["acceleratez"])


        oldAX = state["acceleratex"]
        oldAY = state["acceleratey"]
        updateAngle= math.fabs(self.float * math.pi)
        scope0,scope1=[],[]
        lu0,lu1=[],[]
        if self.orient in [L,UL,DL]:
            if vAngle>=0:
                scope0.append([-math.pi, vAngle - math.pi])
                scope0.append([vAngle, math.pi])
                lu0.append([max(aAngle - updateAngle,vAngle-2*math.pi),
                            min(aAngle + updateAngle,vAngle-math.pi)])
                lu0.append([max(aAngle - updateAngle,vAngle),
                            min(aAngle + updateAngle,vAngle+math.pi)])

                scope1.append([vAngle-math.pi,vAngle])
                lu1.append([aAngle + updateAngle,
                            aAngle +updateAngle
                            ])

            else:
                scope0.append([vAngle, vAngle+math.pi])
                lu0.append([max(aAngle -updateAngle,vAngle),
                            min(aAngle + updateAngle,vAngle+math.pi)])

                scope1.append([vAngle+math.pi,math.pi])
                scope1.append([- math.pi, vAngle])
                lu1.append([aAngle -updateAngle,
                            aAngle - updateAngle
                            ])
                lu1.append([aAngle + updateAngle,
                            aAngle +updateAngle
                            ])

        elif self.orient in [R, UR,DR]:
            if vAngle >= 0:
                scope0.append([vAngle-math.pi, vAngle])
                lu0.append([max(aAngle - updateAngle, vAngle-math.pi),
                            min(aAngle + updateAngle, vAngle)])

                scope1.append([vAngle, math.pi])
                scope1.append([- math.pi, vAngle-math.pi])
                lu1.append([aAngle -updateAngle,
                            aAngle - updateAngle
                            ])
                lu1.append([aAngle - updateAngle,
                            aAngle - updateAngle
                            ])


            else:
                scope0.append([ vAngle+math.pi,math.pi])
                scope0.append([-math.pi,vAngle])
                lu0.append([max(aAngle - updateAngle, vAngle +math.pi),
                            min(aAngle + updateAngle, vAngle+2*math.pi)])
                lu0.append([max(aAngle -updateAngle, vAngle-math.pi),
                            min(aAngle + updateAngle, vAngle)])

                scope1.append([vAngle, vAngle+math.pi])
                lu1.append([aAngle - updateAngle,
                            aAngle - updateAngle
                            ])
        updateAXY = math.fabs(math.sqrt(self.max ** 2 + self.max ** 2) * self.float)
        oldAXYNorm = math.sqrt(oldAX ** 2 + oldAY ** 2)
        newAXYNorm = randomMinMax(min(0, oldAXYNorm - updateAXY),
                                  max(oldAXYNorm + updateAXY, self.max * math.sqrt(2)))
        if self.orient not in [U,D]:
            index = inScopes(aAngle, scope0)
            if index != -1:

                angle = randomMinMax(lu0[index][0], lu0[index][1])

            else:
                index = inScopes(aAngle, scope1)

                if index != -1:
                    angle = randomMinMax(lu1[index][0], lu1[index][1])

                else:

                    raise Exception("The scope of aAngle is wrong")

            angle = angle2pi(angle)
            signX = getXSignByAngle(angle)
            tan = math.tan(angle)
            nextState["acceleratex"] = abs(math.sqrt((newAXYNorm ** 2) / (1 + tan ** 2))) * signX
            nextState["acceleratey"] = nextState["acceleratex"] * tan
        else:
            ##acceleratex以及acceleratey为0

            angle = math.atan2(initStatus["velocityy"], initStatus["velocityx"])
            signX = getXSignByAngle(angle)
        ##Up和Down没有渐进
            tan = math.tan(angle)
            nextState["acceleratex"] = abs(math.sqrt((newAXYNorm ** 2 ) / (1 + tan ** 2))) * signX
            nextState["acceleratey"] = nextState["acceleratex"] * tan
        
        return nextState



    def _checkOrient(self,orient):
        if orient<0 or orient>7:
            raise Exception("The orient should be in [0,7]\n"+self.orientString())


    def orientString(self):
        return "0:Left" \
               "1:Up" \
               "2:Right" \
               "3:Down" \
               "4:UL" \
               "5:DL" \
               "6:UR" \
               "7:DR"

class CONSTAccelerateMaker(VarAccelerateMaker):
    def __init__(self,orient,max,float=0.1,initState=None,lenth=None,tracer=None):
        """scheduler接受StateDict，或者dict为输入，并输出下一时刻的加速度"""
        self._checkOrient(orient)
        self.orient=orient
        self.max = max
        self.float=float
        super(CONSTAccelerateMaker,self).__init__(initState,lenth,tracer)

    def _scheduler(self,state):
        initStatus=self.tracer[-1]
        nextState=state.zero()

        nextState["acceleratez"] = 0
        nextState["acceleratex"] = 0
        nextState["acceleratey"] = 0

        return nextState

    def _checkOrient(self,orient):
        if orient<0 or orient>7:
            raise Exception("The orient should be in [0,7]\n"+self.orientString())

    def orientString(self):
        return "0:CONSTSTRAIGHT" \
               
##2.1圆心常加速度运动模型
# class NormalAccelerateMaker(VarAccelerateMaker):
#
#     def __init__(self,initState,lenth=None,tracer=None):
#
#
#         super(NormalAccelerateMaker,self).__init__(self.f,initState,lenth,tracer)
#
#
#     def _produce(self):
#
#         res=super(NormalAccelerateMaker, self)._produce()
#         return res
#
#     def f(self,state):
#         axis = normalize(rotate90cc(self.tracer.getStateDict(-1,["velocityx","velocityy","velocityz"]).toNP()))
#         newState = state.norm() * axis
#         return dict2statedict({"acceleratex": newState[0], "acceleratey": newState[1], "acceleratez": newState[2]})
#
#
# ##2.2固定半径的圆周运动
# class CircleAccelerateMaker(NormalAccelerateMaker):
#
#     def __init__(self,raidus,lenth=None,tracer=None):
#         self.radius=raidus
#         initStateList=self.tracer.getStateDict(-1,["acceleratex","acceleratey","acceleratez"]).norm()**2/raidus
#         initState=dict2statedict({"acceleratex": initStateList[0], "acceleratey": initStateList[1], "acceleratez": initStateList[2]})
#
#         super(CircleAccelerateMaker,self).__init__(initState,lenth,tracer)
#
#
#


