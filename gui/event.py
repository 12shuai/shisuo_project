import collections


class EventBus:
    def __init__(self,name):
        self.name=name
        self.cache=collections.defaultdict(list)


    def register(self,name,func=None):
        if func is None:
            def wrap(_func):
                self.cache[name].append(_func)
                return _func
            return wrap
        else:
            self.cache[name].append(func)


    def deRegister(self,name):
        self.cache[name].clear()


    def submit(self,name,*args,**kwargs):
        for func in self.cache[name]:
            func(*args,**kwargs)
