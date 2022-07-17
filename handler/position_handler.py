from .handler import Image3DHandler,CSVHandler,Handler,DISPLAY,SAVE
import os

class PositionHandler(Image3DHandler):
    def __init__(self, mode=DISPLAY, save_path=None, start_cfg="ro-", inter_cfg="bo-"):
        super(PositionHandler,self).__init__(mode, save_path, start_cfg,inter_cfg)


    def _handle(self,recorder):
        for idx,state in enumerate(recorder):
            if idx == 0:
                self.ax.plot(state["positionx"], state["positiony"], state["positionz"], self.startCfg)
                self.ax.set_xlabel('x(m)')
                self.ax.set_ylabel('y(m)')
                self.ax.set_zlabel('z(m)')
            else:

                self.ax.plot(state["positionx"], state["positiony"], state["positionz"],self.interCfg)
                self.ax.set_xlabel('x(m)')
                self.ax.set_ylabel('y(m)')
                self.ax.set_zlabel('z(m)')




class DirCSVPositionHandler(Handler):
    def __init__(self, dir,csv_name="csv",image_name="image",image_suffix=".jpg",mode=SAVE,start_cfg="ro-", inter_cfg="bo-"):
        super(DirCSVPositionHandler,self).__init__()
        self.dir=dir
        self.imageDir = image_name
        self.image_suffix=image_suffix if image_suffix[0]=="." else "."+image_suffix
        self.csvDir=csv_name


        self.imageOpt={
            "mode":mode,
            "start_cfg":start_cfg,
            "inter_cfg":inter_cfg
        }

    def _handle(self,recorder):
        self._create_dict()
        prefix=self._get_file_name()

        csvHanlder=CSVHandler(os.path.join(self.dir,self.csvDir,prefix+".csv"))
        imageHandler=PositionHandler(**self.imageOpt,save_path=os.path.join(self.dir,self.imageDir,prefix+self.image_suffix))

        csvHanlder.handle(recorder)
        imageHandler.handle(recorder)




    def _get_file_name(self):
        file_names=os.listdir(os.path.join(self.dir,self.csvDir))

        return str(len(file_names))


    def _create_dict(self):
        try:
            os.makedirs(os.path.join(self.dir,self.csvDir),exist_ok=True)
            os.makedirs(os.path.join(self.dir, self.imageDir), exist_ok=True)

        except Exception as e:
            raise e