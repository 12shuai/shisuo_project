import numpy as np
import numpy.linalg as linalg
from status import StatusDict
from math import cos,sin
"""可以返回工厂"""
def rotate2Vec(src,dst,theta):
    """vector逆时时针旋转degree(单位为度)"""
    res=src.copy()
    vec=src.toNp()
    dst= dst.toNp()
    axis=np.cross(vec,dst)
    if linalg.norm(axis)<0.001:
        return src
    return res.fromNp(rotateByAxis(vec,axis,theta))

def rotateByAxis(vector,axis,theta):
    """vector逆时时针旋转degree(单位为度)"""

    rx,ry,rz=axis/linalg.norm(axis)
    rM=np.array([
        [(rx**2)*(1-cos(theta))+cos(theta),rx*ry*(1-cos(theta))-rz*sin(theta),rx*rz*(1-cos(theta))+ry*sin(theta)],
        [rx*ry*(1-cos(theta))+rz*sin(theta),(ry**2)*(1-cos(theta))+cos(theta),ry*rz*(1-cos(theta))-rx*sin(theta)],
        [rx*rz*(1-cos(theta))-ry*sin(theta),ry*rz*(1-cos(theta))+rx*sin(theta),(rz**2)*(1-cos(theta))+cos(theta)]
        ])

    return rotateByMatrix(vector,rM)



# def rotate90cc(vector,axis):
#     return rotateByAxis(vector,axis,90)
#
# def rotate90c(vector,axis):
#     return rotateByAxis(vector,axis,90,orient=CLOCK)



def rotateByMatrix(vector,matrix):
    return np.matmul(matrix,vector)






def rotateByEuler(vector,*eulerAngle):
    pass

def rotateByQuater(vector,*quater):
    pass


def normalize(vector):
    return linalg.norm(vector)