import math
PI=math.pi



def getSign(value):
    return value/math.fabs(value)

def angle2pi(angle):
    if -PI<angle<PI:
        return angle
    while angle>PI:
        angle-=2*PI
    while angle<-PI:
        angle+=2*PI
    return angle



def getXSignByAngle(angle):
    if PI/2>angle>-PI/2:
        return 1
    else:
        return -1


def getYSignByAngle(angle):
    if PI / 2 > angle >0:
        return 1
    else:
        return -1
