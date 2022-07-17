import sys
import os

from time_series_forecasting_main.time_series_forecasting import my_model
from time_series_forecasting_main.time_series_forecasting.my_training import NormalizeTransform
from gui.model_utils.vit_evaluation import evaluate as vit_evaluate
from gui.event import EventBus

from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from PyQt5.Qt import QThread
from PyQt5.QtGui import QPixmap
from gui.mainwindow import *
import pandas as pd
import numpy as np
import torch





if os.getcwd().endswith("gui"):
    os.chdir(os.path.join(os.getcwd(),".."))
CLASSES_NAMES=['SOMERSAULT', 'DOWN_RIGHT', 'SPIRAL', 'SHARPTURN_RIGHT', 'LEFT', 'DOWN_LEFT', 'DOWN', 'UP_RIGHT', 'UP', 'STRAIGHT', 'HALF_SOMERSAULT', 'SERPENTINE', 'UP_LEFT', 'RIGHT', 'SHARPTURN_LEFT']
DATA_PATH=os.path.join(r"time_series_forecasting_main","train_val_split")


PRETRAINED_DICT={
    "transformer":os.path.join("time_series_forecasting_main","models","ts_views_models","transformer_split.ckpt")
}
MODEL_DICT={
    "transformer":my_model.TimeSeriesForcasting(
            n_encoder_inputs=6,
            classes_num=len(CLASSES_NAMES),
            lr=1e-5,
            dropout=0.1)
}
TRANSFORM_DICT={
    "transformer":NormalizeTransform()
}

EVAL_DICT={
    "transformer":vit_evaluate
}

def loadCKPT():
    device="cuda" if torch.cuda.is_available() else "cpu"
    for k in MODEL_DICT.keys():
        MODEL_DICT[k].load_state_dict(torch.load(PRETRAINED_DICT[k],map_location=device)["state_dict"])


class EvalThread(QThread):
    signal_text_display = QtCore.pyqtSignal(str)
    def __init__(self,parent):
        super().__init__()
        self.parent=parent

    # 开启线程后默认执行
    def run(self):
        modelName = self.parent.comboBox.currentText().lower()
        if not modelName:
            return
        model,transforms,evalFunc=self.parent.modelDict[modelName],self.parent.transformDict[modelName],self.parent.evalDict[modelName]

        total_loss, total_recall, total_precision=evalFunc(self.parent.curDir,transforms,model,self.parent.BATCH,self.parent.classNames)

        self.render(total_loss, total_recall, total_precision)

    def render(self,total_loss, total_recall, total_precision):
        def _renderMap(map,prefix):
            res=prefix
            for k,v in map.items():
                res+=("\n"+k+":"+str(v))
            return res
        text1,text2,text3=_renderMap(total_loss,"1.loss"),_renderMap(total_recall,"2.recall"),\
                          _renderMap(total_precision,"3.precision")
        text="\n\n\n".join([text1,text2,text3])

        self.signal_text_display.emit(text)

class MyWindow(QMainWindow, Ui_MainWindow):
    IMAGE_EXT = ".jpg"
    BATCH=1

    def __init__(self, model_dict,transform_dict,eval_dict,file_path,class_names,parent=None):
        super(MyWindow, self).__init__(parent)
        self.filePath=file_path
        self.transformDict=transform_dict
        self.evalDict=eval_dict
        self.classNames=class_names
        self.transform=None

        self.curTrace,self.curTraceLabel=None,None
        self.curTraceFile,self.curImageFile,self.curDir=None,None,None
        #1.布局
        self.setupUi(self)
        self.reset()

        #2.预加载权重与模型
        self.modelDict=model_dict

        #3.绑定激活函数
        self._bind()

        #4.绑定线程
        self.evalThread=EvalThread(self)
        self._bind_signal()


        ##5.绑定事件
        self.singleEventBus=EventBus("single")
        self.multiEventBus=EventBus("multi")
        self.registerEventBus()

    def reset(self):
        self.detectOnceButton.setEnabled(False)
        self.detectMultiButton.setEnabled(False)

    def _bind(self):
        self.singleButton.clicked.connect(self.chooseCurTrace)
        self.detectOnceButton.clicked.connect(self.detectOnce)
        self.multiButton.clicked.connect(self.chooseBatchDir)
        self.detectMultiButton.clicked.connect(self.detectAll)

    def _bind_signal(self):
        self.evalThread.signal_text_display.connect(self.signal1)

    def signal1(self,msg):
        self.multiEventBus.submit("detectMultiDone")
        self.clearAndRenderText(msg)


    def chooseCurTrace(self):
        try:
            self.curTraceFile=self.chooseFile()
            self.curImageFile=self.getCurImagePath(self.curTraceFile)
            self.curTrace,self.curTraceLabel= self.getTraceAndLabel(self.curTraceFile)
            self.showTraceAndLabel()
            self.singleEventBus.submit("waitDetectOnce")
        except Exception as e:
            self.singleEventBus.submit("fileWrong")


    def chooseFile(self):
        file_name,_ = QFileDialog.getOpenFileName(self, '选择状态序列的csv文件', directory=self.filePath, filter="All Files(*);;Csv Files(*.csv)")
        file_name=file_name.replace(r"/",os.path.sep)

        return file_name

    def getCurImagePath(self,file_name):
        dir_name,name=os.path.split(file_name)
        img_name=os.path.splitext(name)[0]+self.IMAGE_EXT
        img_path=os.path.join(os.path.dirname(dir_name),"image",img_name)
        return img_path

    def getTraceAndLabel(self,file):
        try:
            label = self.classNames.index(file.split(os.path.sep)[-3])
        except:
            return None,None
        data = np.array(pd.read_csv(file))
        data = torch.tensor(data, dtype=torch.float)
        label= torch.tensor(label, dtype=torch.long)

        return data, label

    def showTraceAndLabel(self):
        if self.curImageFile is None:
            return
        pixmap = QPixmap(self.curImageFile)  # 按指定路径找到图片
        self.traceDisplayer.setPixmap(pixmap)  # 在label上显示图片
        self.traceDisplayer.setScaledContents(True)  # 让图片自适应label大小
        self.GTLabel.setText(self.label2Name(self.curTraceLabel))



    def label2Name(self,label):
        if isinstance(label,torch.Tensor):
            label=int(label.cpu().item())
        return self.classNames[label]

    def detectOnce(self):
        try:
            if self.curTrace is None or self.curTraceLabel is None:
                return
            modelName=self.comboBox.currentText()
            if not modelName:
                return
            model,transform=self.modelDict[modelName.lower()],self.transformDict[modelName.lower()]
            data,label=self.curTrace,self.curTraceLabel
            if torch.cuda.is_available():
                data,label,model=data.cuda(),label.cuda(),model.cuda()
            data,label=self.preprocess(data,label,transform)
            pred,res=model.predict(data,label)
            self.revealPredictResult(self.label2Name(pred),"检测正确" if res else "检测错误")
            self.singleEventBus.submit("detectOnceDone")
        except:
            self.singleEventBus.submit("parseWrong")

    def preprocess(self,data,label,transform):
        data=transform(data)
        data,label=data.unsqueeze(0),label.unsqueeze(0)
        return data,label

    def revealPredictResult(self,pred,res):
        self.predictLabel.setText(str(pred))
        self.resultLabel.setText(str(res))

    def chooseBatchDir(self):
        self.curDir= QFileDialog.getExistingDirectory(self,"选取文件夹",self.filePath)
        self.curDir=self.curDir.replace(r"/", os.path.sep)
        if self.curDir and os.path.exists(self.curDir):
            self.multiEventBus.submit("waitMultiDetect")
        else:
            pass


    def detectAll(self):
        if self.curDir is None or not self.checkDir(self.curDir):
            self.multiEventBus.submit("dirWrong")
            return
        try:
            self.multiEventBus.submit("detectMultiStart")
            self.evalThread.start()
            return
        except:
            self.multiEventBus.submit("detectMultiWrong")
            return




    def checkDir(self,dirName):
        dirs=os.listdir(dirName)
        for d in dirs:
            if d not in self.classNames:
                return False

        return True

    def clearAndRenderText(self,text):
        self.BatchResultDisplay.clear()
        self.renderText(text)

    def renderText(self,text):
        self.BatchResultDisplay.setPlainText(text)


    def registerEventBus(self):
        self.registerSingleEventBus()
        self.registerMultiEventBus()

    def registerSingleEventBus(self):
        self.singleEventBus.register("fileWrong",self.fileWrongHandler)
        self.singleEventBus.register("waitDetectOnce", self.waitDetectOnceHandler)
        self.singleEventBus.register("parseWrong", self.parseWrongHandler)
        self.singleEventBus.register("detectOnceDone", self.detectOnceDoneHandler)

    def registerMultiEventBus(self):
        self.multiEventBus.register("dirWrong",self.dirWrongHandler)
        self.multiEventBus.register("waitMultiDetect", self.waitDetectMultiHandler)
        self.multiEventBus.register("detectMultiStart", self.detectMultiStartHandler)
        self.multiEventBus.register("detectMultiWrong", self.detectMultiWrongHandler)
        self.multiEventBus.register("detectMultiDone", self.detectMultiDoneHandler)


    def fileWrongHandler(self):
        self.messageLabel.setText("输入的文件有问题，请输入csv格式文件")
        self.detectOnceButton.setEnabled(False)

    def parseWrongHandler(self):
        self.messageLabel.setText("输入的文件内容有误")
        self.detectOnceButton.setEnabled(False)

    def waitDetectOnceHandler(self):
        self.messageLabel.setText("请检测")
        self.detectOnceButton.setEnabled(True)

        self.predictLabel.setText("还未检测")
        self.resultLabel.setText("还未检测")

    def detectOnceDoneHandler(self):
        self.messageLabel.setText("检测完毕")

    def dirWrongHandler(self):
        self.messageLabel.setText("输入的文件夹的组织关系有误")
        self.detectMultiButton.setEnabled(False)

    def waitDetectMultiHandler(self):
        self.messageLabel.setText("等待批量检测")
        self.detectMultiButton.setEnabled(True)

    def detectMultiStartHandler(self):
        self.messageLabel.setText("批量检测开始")
        self.multiButton.setEnabled(False)
        self.detectMultiButton.setEnabled(False)

    def detectMultiWrongHandler(self):
        self.messageLabel.setText("批量检测出错")
        self.multiButton.setEnabled(True)
        self.detectMultiButton.setEnabled(False)

    def detectMultiDoneHandler(self):
        self.messageLabel.setText("检测完毕")
        self.multiButton.setEnabled(True)
        self.detectMultiButton.setEnabled(True)



if __name__ == '__main__':
    loadCKPT()
    app = QApplication(sys.argv)
    myWin = MyWindow(MODEL_DICT,TRANSFORM_DICT,EVAL_DICT,DATA_PATH,CLASSES_NAMES)
    myWin.show()
    sys.exit(app.exec_())