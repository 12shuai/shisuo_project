import csv

from status import Tracer
import matplotlib.pyplot as plt

DISPLAY = 0
SAVE = 1
BOTH = 2


class Handler:

    def __init__(self):
        pass

    def handle(self,recoder):
        if not isinstance(recoder,Tracer):
            raise TypeError("Input should be Tracer type")
        self._handle(recoder)

    def _handle(self,recorder):
        raise NotImplementedError()




class CSVHandler(Handler):
    def __init__(self,path):
        self.path=path
        self.header=None
        super(CSVHandler, self).__init__()


    def _handle(self,recorder):
        def _getColNames(stateName):
            headers = []
            for name in stateName:
                headers.append(name)
            return headers

        with open(self.path,"w") as f:
            f_csv = csv.writer(f)
            if not self.header:
                self.header = _getColNames(recorder.stateName)
            f_csv.writerow(self.header)
            for state in recorder:
                insert = []
                for name in self.header:
                    insert.append(state[name])
                f_csv.writerow(insert)



class Image3DHandler(Handler):


    def __init__(self, mode=DISPLAY, save_path=None, start_cfg="ro-", inter_cfg="bo-"):
        if mode > BOTH:
            raise Exception("mode should in [0,1,2], [DISPLAY,SAVE,BOTH]")
        self.mode = mode
        if mode != DISPLAY:
            if not save_path:
                raise Exception("If you want to save figure, you should give the save_path")
            self.path = save_path

        self.startCfg = start_cfg
        self.interCfg = inter_cfg

        super(Image3DHandler, self).__init__()


    def handle(self,recorder):
        self._resetFig()
        super(Image3DHandler,self).handle(recorder)
        if self.mode!=SAVE:
            # self.fig.show()
            plt.show()
        if self.mode!=DISPLAY:
            self.fig.savefig(self.path)

    def set_save_path(self,path):
        self.path=path


    def _resetFig(self):
        self.fig=plt.figure()
        self.ax=self.fig.gca(projection='3d')














