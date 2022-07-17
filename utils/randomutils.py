import random
import math
from math import pi as PI

MINI=-65535
MAXI=65535

def randomIntMinMax(min=None,max=None):
    if min is None and max is None:
        min,max=MINI,MAXI
    elif min is None:
        min=MINI
    elif max is None:
        max=MAXI
    return random.randint(min,max)




def randomMinMax(min=None,max=None):
    if min is None and max is None:
        min,max=MINI,MAXI
    elif min is None:
        min=MINI
    elif max is None:
        max=MAXI
    return random.random() * (max - min) + min


def randomLeft(angle):
    if 0<angle<PI/2 or PI/2<angle<PI:
        random1=randomMinMax(angle,PI)
        random2=randomMinMax(-PI,-PI+angle)
        return random.choice([random1, random2])

    elif -PI/2<angle<0 or -PI<angle<-PI/2:
        random1=randomMinMax(angle,PI+angle)
        return random1

    else:
        raise Exception("angle should in [-π，π]")


def randomRight(angle):
    if -PI/2<angle<0 or -PI<angle<-PI/2:
        random1=randomMinMax(angle+PI,PI)
        random2=randomMinMax(-PI,angle)
        return random.choice([random1, random2])

    elif 0<angle<PI/2 or PI/2<angle<PI:
        random1=randomMinMax(angle-PI,angle)
        return random1

    else:
        raise Exception("angle should in [-π，π]")



