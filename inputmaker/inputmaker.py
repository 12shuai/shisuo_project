from status import ELExceptionRaise,dict2statedict
from itertools import chain

class InputComposer:
    def __init__(self,*inputMaker):
        self.nameSet = None
        try:
            self.check(inputMaker)
        except Exception as e:
            raise e
        self.inputMaker=inputMaker


    def check(self,*inputMaker):
        if not inputMaker:
            raise Exception("Composer should have at least one InputMaker")

        for inp in inputMaker:
            if not isinstance(inp,InputMaker):
                raise TypeError("The input should be InputMaker type")
            if not self.nameSet:
                self.nameSet=inp.nameSet
                continue
            if self.nameSet!=inp.nameSet:
                raise Exception("The inputs(InputMaker type) should have same nameSet")

        return

    def __iter__(self):
        return chain.from_iterable(self.inputMaker)



class InputMaker:
    INFINITE=0
    FINITE=1
    def __init__(self,nameSet,initState=None,lenth=None,tracer=None,**kwargs):
        if not lenth:
            self.mode=self.INFINITE
        else:
            self.mode=self.FINITE
            self.lenth=lenth
        self.nameSet=nameSet
        self.tracer=tracer
        self.initState=None
        if initState:

            try:
                self.checkState(initState)
                self.initState = dict2statedict(initState)

            except Exception as e:
                raise e



    def setTracer(self,tracer):
        self.tracer=tracer

    def __iter__(self):
        return self

    def produce(self):

        res = self._produce()

        self.checkState(res)

        return dict2statedict(res)

    def _produce(self):
        raise NotImplementedError()

    def __next__(self):
        if self.mode:
            if self.lenth:

                self.lenth-=1
                return self.produce()
            else:

                raise StopIteration()

        else:
            try:
                while True:
                    return self.produce()
            except InterruptedError:
                raise StopIteration()
            except Exception as e:
                raise e

    def checkState(self,state):

        ELExceptionRaise(self.nameSet,state,"State")







