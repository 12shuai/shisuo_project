from scheduler import Scheduler
from status import PVStateTransferFunc
from inputmaker import *
from handler import *

##1.基本事件参数设置
OUTPUT_DIR="output/"

##2.采样步长与采样间隔
INTERVAL=0.08
LENTH=20

##3.最大位移，速度与加速度限制
MAX_VELOCITY=240
MAX_ACC=20
MIN_X,MAX_X=-800000,800000 ##单位以及设置XYZ方向的condition
MIN_Y,MAX_Y=-800000,800000
MIN_Z,MAX_Z=8000,18000

##4.基本事件的符号设定
# STRAIGHT="S"##直飞
# LEFT="L"##左转
# RIGHT="R"##右转
# UP="U"##爬升
# DOWN="D"##俯冲
# UP_LEFT="UL"##左爬升
# DOWN_LEFT="DL"##左俯冲
# UP_RIGHT="UR"##右爬升
# DOWN_RIGHT="DR"##右俯冲

# HORIZONTAL_STRAIGHT="HS" ##水平直飞
# UNIFORM_STRAIGHT="US"##匀速前飞
# DECELERATE_STRAIGHT="DAS"##减速前飞
# ACCELERATE_STRAIGHT="AS"##加速前飞

##4.机动事件的符号设定
SOMERSAULT="U+D"##筋斗
HALF_SOMERSAULT="U+S"##半筋斗
SHARPTURN_LEFT = "L+S"##左急转
SHARPTURN_RIGHT = "R+S"##右急转
SPIRAL = "T+T+T+T"##盘旋
SERPENTINE = "L+R+L+R"##蛇形机动

##5.状态的范围
COND_STATUS={
   "positionx":[MIN_X,MAX_X],
   "positiony":[MIN_Y,MAX_Y],
    "positionz":[MIN_Z,MAX_Z],
    "velocityx":[-MAX_VELOCITY,MAX_VELOCITY],
    "velocityy":[-MAX_VELOCITY,MAX_VELOCITY],
   "velocityz":[-MAX_VELOCITY,MAX_VELOCITY]
}

COND_STATUS2={
   "positionx":[MIN_X,MAX_X],
   "positiony":[MIN_Y,MAX_Y],
    "positionz":[MIN_Z,MAX_Z],
    "velocityx":[-MAX_VELOCITY,MAX_VELOCITY],
    "velocityy":[-MAX_VELOCITY,MAX_VELOCITY],
   "velocityz":[-MAX_VELOCITY,MAX_VELOCITY]
}

##6.初始状态的范围
ENV_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z/2,MAX_Z*3/4],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/20,MAX_VELOCITY/20]
   #  "velocityz":[0,0]
}

STRAIGHT_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z/2,MAX_Z*3/4],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[0,0.01]
   #  "velocityz":[0,0]
}
LEFT_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z/2,MAX_Z*3/4],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[0,1]
   #  "velocityz":[0,0]
}
RIGHT_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z/2,MAX_Z*3/4],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}
UP_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MIN_Z*9/8,MAX_Z*3/4],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}
DOWN_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z*3/4,MAX_Z],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}
UP_LEFT_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MIN_Z*9/8,MAX_Z/2],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}
DOWN_LEFT_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z*3/4,MAX_Z],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}
UP_RIGHT_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MIN_Z*9/8,MAX_Z/2],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}
DOWN_RIGHT_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z*3/4,MAX_Z],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}

SPIRAL_STATUS={
   "positionx":[MIN_X/8,MAX_X/8],
   "positiony":[MIN_Y/8,MAX_Y/8],
    "positionz":[MAX_Z*3/4,MAX_Z],
    "velocityx":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
    "velocityy":[-MAX_VELOCITY/2,MAX_VELOCITY/2],
   "velocityz":[-MAX_VELOCITY/200,MAX_VELOCITY/200]
   #  "velocityz":[0,0]
}


##7.初始条件
# ###7.1.1直线飞行的初始条件
# STRAIGHT_INPUTMAKER=[
#     CONSTAccelerateMaker,{
#         "orient":0,
#        "max":MAX_ACC,
#         "lenth":400
#    }
# ]
#
# # STRAIGHT_HANDLERS=[PositionHandler()]
# ###7.1.2直线飞行的输出形式
# STRAIGHT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"STRAIGHT")]
#
#
# ###7.2.1左转的初始条件
# LEFT_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":0,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
#
# # LEFT_HANDLERS=[PositionHandler()]
# ###7.2.2左转的输出形式
# LEFT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"LEFT")]
#
#
# ###7.3.1右转的初始条件
# RIGHT_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":2,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
# # RIGHT_HANDLERS=[PositionHandler()]
# ###7.3.2右转的输出形式
# RIGHT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"RIGHT")]
#
#
#
# ###7.4.1爬升的初始条件
# UP_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":1,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
# # UP_HANDLERS=[PositionHandler()]
# ###7.4.2爬升的输出形式
# UP_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"UP")]
#
# ###7.5.1俯冲的初始条件
# DOWN_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":3,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
# # DOWN_HANDLERS=[PositionHandler()]
# ###7.5.2俯冲的形式
# DOWN_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"DOWN")]
#
#
# ###7.6.1左爬升的初始条件
# UP_LEFT_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":4,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
# # UP_LEFT_HANDLERS=[PositionHandler()]
# ###7.6.2左爬升的输出形式
# UP_LEFT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"UP_LEFT")]
#
#
#
# ###7.7.1左俯冲的初始条件
# DOWN_LEFT_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":5,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
# # DOWN_LEFT_HANDLERS=[PositionHandler()]
# ###7.7.2左俯冲的输出形式
# DOWN_LEFT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"DOWN_LEFT")]
#
# ###7.8.1.右爬升的初始条件
# UP_RIGHT_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":6,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
#
# # UP_RIGHT_HANDLERS=[PositionHandler()]
# ###7.8.2.右爬升的输出形式
# UP_RIGHT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"UP_RIGHT")]
#
#
#
# ###7.9.1.右俯冲的初始条件
# DOWN_RIGHT_INPUTMAKER=[OrientTurnAccelerateMaker,{
#        "orient":7,
#        "max":MAX_ACC,
#         "lenth":400
#    }]
#
# # DOWN_RIGHT_HANDLERS=[PositionHandler()]
# ###7.9.2.右俯冲的输出形式
# DOWN_RIGHT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"DOWN_RIGHT")]

##7.机动事件初始条件和输出形式
###7.1筋斗
SOMERSAULT_INPUTMAKER=[OrientTurnAccelerateMaker,{
       "orient":1,
       "max":MAX_ACC,
        "lenth":200
   },OrientTurnAccelerateMaker,{
        "orient":3,
       "max":MAX_ACC,
        "lenth":200
   }]

SOMERSAULT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"SOMERSAULT")]

###7.2.半筋斗
HALF_SOMERSAULT_INPUTMAKER=[OrientTurnAccelerateMaker,{
       "orient":1,
       "max":MAX_ACC,
        "lenth":200
   },CONSTAccelerateMaker,{
        "orient":0,
       "max":MAX_ACC,
        "lenth":200
   }]

HALF_SOMERSAULT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"HALF_SOMERSAULT")]

###7.3左急转
SHARPTURN_LEFT_INPUTMAKER=[OrientTurnAccelerateMaker,{
       "orient":0,
       "max":MAX_ACC,
        "lenth":200
   },CONSTAccelerateMaker,{
        "orient":0,
       "max":MAX_ACC,
        "lenth":200
   }]

SHARPTURN_LEFT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"SHARPTURN_LEFT")]

###7.4右急转
SHARPTURN_RIGHT_INPUTMAKER=[OrientTurnAccelerateMaker,{
       "orient":2,
       "max":MAX_ACC,
        "lenth":200
   },CONSTAccelerateMaker,{
        "orient":0,
       "max":MAX_ACC,
        "lenth":200
   }]

SHARPTURN_RIGHT_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"SHARPTURN_RIGHT")]

###7.5盘旋
SPIRAL_INPUTMAKER=[OrientTurnAccelerateMaker,{
       "orient":2,
       "max":MAX_ACC,
        "lenth":800},OrientTurnAccelerateMaker,{
       "orient":2,
       "max":MAX_ACC,
        "lenth":800}]

SPIRAL_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"SPIRAL")]

###7.6蛇形机动
SERPENTINE_INPUTMAKER=[OrientTurnAccelerateMaker,{
       "orient":0,
       "max":MAX_ACC*2,
        "lenth":400},OrientTurnAccelerateMaker,{
       "orient":2,
       "max":MAX_ACC*2,
        "lenth":400},OrientTurnAccelerateMaker,{
       "orient":0,
       "max":MAX_ACC*2,
        "lenth":400},OrientTurnAccelerateMaker,{
       "orient":2,
       "max":MAX_ACC*2,
        "lenth":400}]

SERPENTINE_HANDLERS=[DirCSVPositionHandler(OUTPUT_DIR+"SERPENTINE")]

class BaseEventScheduler(Scheduler):
    def __init__(self,condition,setting,interval,lenth):
        super(BaseEventScheduler,self).__init__(condition,setting,lenth,PVStateTransferFunc(),interval)



if __name__ == '__main__':
   # setting={
   #     STRAIGHT:[[STRAIGHT_STATUS,STRAIGHT_INPUTMAKER,STRAIGHT_HANDLERS]],
   #     LEFT:[[LEFT_STATUS,LEFT_INPUTMAKER,LEFT_HANDLERS]],
   #     RIGHT:[[RIGHT_STATUS,RIGHT_INPUTMAKER,RIGHT_HANDLERS]],
   #     UP:[[UP_STATUS,UP_INPUTMAKER,UP_HANDLERS]],
   #     DOWN:[[DOWN_STATUS,DOWN_INPUTMAKER,DOWN_HANDLERS]],
   #     UP_LEFT:[[UP_LEFT_STATUS,UP_LEFT_INPUTMAKER,UP_LEFT_HANDLERS]],
   #     DOWN_LEFT:[[DOWN_LEFT_STATUS,DOWN_LEFT_INPUTMAKER,DOWN_LEFT_HANDLERS]],
   #     UP_RIGHT:[[UP_RIGHT_STATUS,UP_RIGHT_INPUTMAKER,UP_RIGHT_HANDLERS]],
   #     DOWN_RIGHT:[[DOWN_RIGHT_STATUS,DOWN_RIGHT_INPUTMAKER,DOWN_RIGHT_HANDLERS]]
   # }
   # scheduler1 = BaseEventScheduler(COND_STATUS, setting, interval=INTERVAL, lenth=LENTH)
   # scheduler1.forward()

   setting2 = {
      SOMERSAULT:[[UP_STATUS,SOMERSAULT_INPUTMAKER,SOMERSAULT_HANDLERS]],
      HALF_SOMERSAULT:[[UP_STATUS,HALF_SOMERSAULT_INPUTMAKER,HALF_SOMERSAULT_HANDLERS]],
      #SHARPTURN_LEFT:[[LEFT_STATUS,SHARPTURN_LEFT_INPUTMAKER,SHARPTURN_LEFT_HANDLERS]],
      #SHARPTURN_RIGHT:[[RIGHT_STATUS,SHARPTURN_RIGHT_INPUTMAKER,SHARPTURN_RIGHT_HANDLERS]]
   }

   setting4 = {
      SERPENTINE:[[LEFT_STATUS,SERPENTINE_INPUTMAKER,SERPENTINE_HANDLERS]]
   }


   scheduler2=BaseEventScheduler(COND_STATUS,setting2,interval=INTERVAL,lenth=LENTH)
   scheduler2.forward2()
   scheduler4=BaseEventScheduler(COND_STATUS2,setting4,interval=INTERVAL,lenth=LENTH)
   scheduler4.forward4()

