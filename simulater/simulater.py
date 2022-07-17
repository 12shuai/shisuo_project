from status import Condition,Tracer,Stepper,ELExceptionRaise

class Simulater:

    def __init__(self,initStatus,condition,stepperFunc,interval,inputMaker,handler=None):
        self._init(initStatus,condition,stepperFunc,interval,inputMaker)
        if isinstance(handler,list):
            self.handlers=handler
        else:
            self.handlers=[]
            self.handlers.append(handler)

    def reset(self):
        self.recorder.reset()

    def setInitState(self,initStatus):
        self.initStatus=initStatus

    def setInterval(self,interval):
        self.stepper.setInterval(interval)

    def addHandler(self,*handler):
        self.handlers.extend(handler)

    def setHandler(self,handler):
        self.handlers=[handler] if not isinstance(handler,list) else handler


    def _init(self,initStatus,condition,stepperFunc,interval,inputMaker):
        self.initStatus = initStatus.copy()
        self.condition=Condition(condition)
        ELExceptionRaise(self.condition.stateName, self.initStatus, "Condition")
        self.stepper = Stepper(stepperFunc, interval)
        ELExceptionRaise(stepperFunc.stateName,self.initStatus,"StepperFunc")
        self.recorder = Tracer(self.initStatus.keys())
        self.recorder.append(self.initStatus)
        ELExceptionRaise(stepperFunc.inputName,inputMaker.nameSet,"InputMaker")

        self.inputMaker = inputMaker
        self.inputMaker.setTracer(self.recorder)
       
    def forward(self):
        state=self.initStatus
        for input in self.inputMaker:
            state=self.stepper(state,input)
            state=self._correct(state)
            self.recorder.append(state)

        self.handle()
        
    def _correct(self,state):

        return self.condition.correct(state)

    def handle(self):
        for handler in self.handlers:
            handler.handle(self.recorder)

    def __call__(self, *args, **kwargs):
        self.forward()

class Simulater2:

    def __init__(self,initStatus,condition,stepperFunc,interval,inputMaker1,inputMaker2,handler=None):
        self._init(initStatus,condition,stepperFunc,interval,inputMaker1,inputMaker2)
        if isinstance(handler,list):
            self.handlers=handler
        else:
            self.handlers=[]
            self.handlers.append(handler)

    def reset(self):
        self.recorder.reset()

    def setInitState(self,initStatus):
        self.initStatus=initStatus


    def setInterval(self,interval):
        self.stepper.setInterval(interval)

    def addHandler(self,*handler):
        self.handlers.extend(handler)

    def setHandler(self,handler):
        self.handlers=[handler] if not isinstance(handler,list) else handler

    def _init(self,initStatus,condition,stepperFunc,interval,inputMaker1,inputMaker2):
        self.initStatus = initStatus.copy()
        self.condition=Condition(condition)
        ELExceptionRaise(self.condition.stateName, self.initStatus, "Condition")
        self.stepper = Stepper(stepperFunc, interval)
        ELExceptionRaise(stepperFunc.stateName,self.initStatus,"StepperFunc")
        self.recorder = Tracer(self.initStatus.keys())
        self.recorder.append(self.initStatus)
        ELExceptionRaise(stepperFunc.inputName,inputMaker1.nameSet,"InputMaker1")

        self.inputMaker1 = inputMaker1
        self.inputMaker1.setTracer(self.recorder)
        self.inputMaker2 = inputMaker2
        self.inputMaker2.setTracer(self.recorder)

    def forward(self):
        state=self.initStatus
        for input in self.inputMaker1:
            state=self.stepper(state,input)
            state=self._correct(state)
            self.recorder.append(state)
        print('1')
        
        for input in self.inputMaker2:
            state=self.stepper(state,input)
            state=self._correct(state)
            if state["velocityz"] > 0:
                state["velocityz"] -= 4
            '''if state["velocityz"] < 0:
                state["velocityz"] += 1'''
            self.recorder.append(state)
        self.handle()

    def _correct(self,state):

        return self.condition.correct(state)

    def handle(self):
        for handler in self.handlers:
            handler.handle(self.recorder)

    def __call__(self, *args, **kwargs):
        self.forward()

class Simulater4:

    def __init__(self,initStatus,condition,stepperFunc,interval,inputMaker1,inputMaker2,inputMaker3,inputMaker4,handler=None):
        self._init(initStatus,condition,stepperFunc,interval,inputMaker1,inputMaker2,inputMaker3,inputMaker4)
        if isinstance(handler,list):
            self.handlers=handler
        else:
            self.handlers=[]
            self.handlers.append(handler)

    def reset(self):
        self.recorder.reset()

    def setInitState(self,initStatus):
        self.initStatus=initStatus


    def setInterval(self,interval):
        self.stepper.setInterval(interval)

    def addHandler(self,*handler):
        self.handlers.extend(handler)

    def setHandler(self,handler):
        self.handlers=[handler] if not isinstance(handler,list) else handler

    def _init(self,initStatus,condition,stepperFunc,interval,inputMaker1,inputMaker2,inputMaker3,inputMaker4):
        self.initStatus = initStatus.copy()
        self.condition=Condition(condition)
        ELExceptionRaise(self.condition.stateName, self.initStatus, "Condition")
        self.stepper = Stepper(stepperFunc, interval)
        ELExceptionRaise(stepperFunc.stateName,self.initStatus,"StepperFunc")
        self.recorder = Tracer(self.initStatus.keys())
        self.recorder.append(self.initStatus)
        ELExceptionRaise(stepperFunc.inputName,inputMaker1.nameSet,"InputMaker1")

        self.inputMaker1 = inputMaker1
        self.inputMaker1.setTracer(self.recorder)
        self.inputMaker2 = inputMaker2
        self.inputMaker2.setTracer(self.recorder)
        self.inputMaker3 = inputMaker3
        self.inputMaker3.setTracer(self.recorder)
        self.inputMaker4 = inputMaker4
        self.inputMaker4.setTracer(self.recorder)

    def forward(self):
        state=self.initStatus
        for input in self.inputMaker1:
            state=self.stepper(state,input)
            self.recorder.append(self._correct(state))
        print('1')
        
        for input in self.inputMaker2:
            state=self.stepper(state,input)
            self.recorder.append(self._correct(state))
        
        for input in self.inputMaker3:
            state=self.stepper(state,input)
            self.recorder.append(self._correct(state))
        
        for input in self.inputMaker4:
            state=self.stepper(state,input)
            self.recorder.append(self._correct(state))
        self.handle()

    def _correct(self,state):

        return self.condition.correct(state)

    def handle(self):
        for handler in self.handlers:
            handler.handle(self.recorder)

    def __call__(self, *args, **kwargs):
        self.forward()
