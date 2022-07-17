import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from einops import rearrange, repeat

class TimeSeriesForcasting(pl.LightningModule):
    def __init__(
        self,
        n_encoder_inputs,
        classes_num,
        channels=512,
        dropout=0.1,
        lr=1e-4,
        method=None ##mean
    ):
        super().__init__()
        self.classes_num=classes_num
        self.method=method
        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout
        
        #1.态势感知部分
        ##1.1可学习的位置编码（需要注意维度大小，必须大于最长时间序列长度）
        self.input_pos_embedding = torch.nn.Embedding(1700, embedding_dim=channels)
        
        ##1.2输入映射
        self.input_projection = Linear(n_encoder_inputs, channels)

        ##1.3编码器层（dim_feedforward为FFN层的中间的扩维大小）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        ##1.4总的编码器
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)

        ##1.5全局查询嵌入v_0
        self.cls_token = nn.Parameter(torch.randn(1,1,channels))
        
        #2.分类网络
        self.classifier=nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels,classes_num)
        )

        self.linear = Linear(channels, 1)

        # self.do = nn.Dropout(p=self.dropout)

        self.cross_loss=nn.CrossEntropyLoss()

    def encode_src(self, src):
        #1.状态序列经过映射，称为嵌入式输入v
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        
        #2.得到位置编码
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        #3.添加位置编码
        src = src_start + pos_encoder

        #4.添加全局查询嵌入（其不用加位置编码，所以在位置编码操作后执行）
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)                # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        cls_tokens=cls_tokens.permute(1, 0, 2)
        src = torch.cat((cls_tokens, src), dim=0)                                         # 将cls_token拼接到patch token中去       (b, 65, dim)
        # src=self.do(src)

        #5.输入到编码器网络
        src = self.encoder(src) ##+ src_start
        src=src.permute(1,0,2)
        if self.method and self.method=="mean":
            src= src.mean(dim=1) 
        else:
            src=src[:, 0]                                                                      # (b, dim)

        return src


    def forward(self, x):
        src = x
        src = self.encode_src(src) ##态势感知模型部分
        
        out = self.classifier(src) ##分类模型部分
        return out


    def predict(self,trace,label):
        self.eval()
        with torch.no_grad():
            pred=self.forward(trace)
        self.train()
        index=torch.argmax(pred).cpu().item()
        return index,torch.argmax(pred)==label

    def training_step(self, batch, batch_idx):
        src, label = batch

        y_hat = self(src)

        loss = self.cross_loss(y_hat, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, label = batch

        y_hat = self(src)

        loss = self.cross_loss(y_hat, label)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, label = batch

        y_hat = self(src)

        loss = self.cross_loss(y_hat, label)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


   
class LSTMTimeSeriesForcasting(pl.LightningModule):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - num_classes: number of classes
        - num_layers: layers of LSTM to stack
    """

    def __init__(self,input_size,hidden_size,num_classes,num_layers,strategy="last",lr=1e-4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, num_classes)
        self.cross_loss=nn.CrossEntropyLoss()
        self.lr=lr
        self.strategy=strategy

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        # b,s, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # x = x.view(b*s, h)
        x = self.forwardCalculation(x)
        #x = x.view(b,s, -1)
        # loss=self.lossFunction(x,label)
        
        if self.strategy=="mean":
            return x.mean(axis=1)
        elif self.strategy=="last":
            return x[:,-1,:]

    def training_step(self, batch, batch_idx):
        src, label = batch

        y_hat = self(src)

        loss = self.cross_loss(y_hat, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, label = batch

        y_hat = self(src)

        loss = self.cross_loss(y_hat, label)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, label = batch

        y_hat = self(src)

        loss = self.cross_loss(y_hat, label)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }

# if __name__ == "__main__":
#     n_classes = 100

#     source = torch.rand(size=(32, 16, 9))
#     target_in = torch.rand(size=(32, 16, 8))
#     target_out = torch.rand(size=(32, 16, 1))

#     ts = TimeSeriesForcasting(n_encoder_inputs=9, n_decoder_inputs=8)

#     pred = ts((source, target_in))

#     print(pred.size())

#     ts.training_step((source, target_in, target_out), batch_idx=1)
