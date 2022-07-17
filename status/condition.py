from .state import Status,StatusDict
from utils import randomMinMax
import math
MINI=-65535
MAXI=65535

class Condition:
    def __init__(self,scopes):
        """
        :param scopes:字典对象，值为列表，代表范围，None表示无界
        """
        self.scopes={}
        for k,v in scopes.items():
            if not isinstance(v,list) and len(v)!=2:
                raise Exception("The value in scopes should be list and len(list) should be 2")
            min,max=v
            if min is None and max is None:
                min, max = MINI, MAXI
            elif min is None:
                min = MINI
            elif max is None:
                max = MAXI
            self.scopes[k]=[min,max]
        self.stateName=list(self.scopes.keys())


    def correct(self,state):
        res = state
        for k, v in self.scopes.items():
            min, max = v
            '''if k in ["velocityx","velocityy"] :
                if min <=  res[k] <= max :
                    continue 
                elif res[k] < min:
                    res[k] = res[k]*math.sqrt( min ** 2 + min ** 2)/math.sqrt( res["velocityx"] ** 2 + res["velocityy"] ** 2)
                else :
                    res[k] = res[k]*math.sqrt( max ** 2 + max ** 2)/math.sqrt( res["velocityx"] ** 2 + res["velocityy"] ** 2)
            else:'''
            if min<=res[k]<=max:
                continue
            elif res[k]<min:
                res[k]=min
            else:
                res[k]=max

        return res


    def randomState(self):
        """
        :return:产生对应范围的StatusDict
        """
        res=StatusDict()
        for k,v in self.scopes.items():
            min,max=v
            res.append(Status(k,randomMinMax(min,max)))


        return res