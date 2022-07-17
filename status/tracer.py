from .state import StatusDict,ELExceptionRaise,Status
def _makeTracer(stateName):
    res={}
    for k in stateName:
        res[k]=[]
    return res

class Tracer:
    def __init__(self,stateName):

        self.stateName=stateName
        self.tracer=_makeTracer(stateName)
        self.curr=0
        self.lenth=0

    def append(self,statusDict):
        ELExceptionRaise(self.stateName,statusDict)

        for k,v in statusDict.items():
            self.tracer[k].append(v)
        self.lenth+=1


    def reset(self):
        self.tracer=_makeTracer(self.stateName)
        self.curr=0
        self.lenth=0

    def __getitem__(self, item):
        return self.getStateDict(-1)

    def getStateDict(self,index,names=None):
        res={}
        if not names:
            for k,v in self.tracer.items():
                res[k]=self.tracer[k][index]
        else:
            if not isinstance(names,list):
                raise TypeError("names should be list type")

            for k in names:
                res[k]=self.tracer[k][index]

        return StatusDict(res)


    def __iter__(self):
        return self

    def __next__(self):
        if self.curr==self.lenth-1:
            self.curr=0
            raise StopIteration()
        res=StatusDict()
        for k,v in self.tracer.items():

            res.append(Status(k,v[self.curr]))
        self.curr+=1
        return res
