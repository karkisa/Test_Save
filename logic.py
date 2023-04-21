import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from base_pipe import base_pipe
import pdb,wandb
import pandas as pd
from sklearn.model_selection import KFold

def collate_fn(batch):
    return tuple(zip(*batch))

class classifier(pl.LightningModule):

    def __init__(self,
                 fold,
                 df,
                 model,
                 ds,
                 bs,
                 num_classes,
                 shuffle = False,
                 num_workers = 8,
                 loss=torch.nn.CrossEntropyLoss(),
                 lr=2e-3,
                 wandb_run=None
                 ):
        
        super().__init__()
        self.train_img_list = df[df['fold']!=fold]['img_list'].values
        self.val_img_list =  df[df['fold']==fold]['img_list'].values
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.wandb_run = wandb_run
        self.model = model
        self.loss = loss
        self.lr = lr
        self.bs = bs
        self.ds = ds
        self.columns=["id", "image", "guess", "truth"]
        for digit in range(num_classes):
            self.columns.append("score_" + str(digit))
        self.test_table = wandb.Table(columns=self.columns)

    def train_dataloader(self) :
        ds = self.ds(self.train_img_list)
        train_loader = DataLoader(dataset=ds,
                                  num_workers=self.num_workers,
                                  shuffle=self.shuffle,
                                  batch_size=self.bs,
                                  collate_fn=collate_fn
                                  )
        return train_loader
    
    def val_dataloader(self):
        ds = self.ds(self.val_img_list)
        val_loader = DataLoader(dataset=ds,
                                num_workers=self.num_workers,
                                shuffle= self.shuffle,
                                batch_size=self.bs,
                                collate_fn=collate_fn
                                )
        return val_loader
    
    def training_step(self,batch,bath_id):
        imgs,targets = batch
        outputs = self.model(torch.stack(list(imgs), dim=0))
        loss_on_step = self.loss(outputs,torch.tensor(targets))
        #log on wandb
        if self.wandb_run:
            self.wandb_run.log({"train" : {"train_loss" : loss_on_step}}, commit = True)
        return loss_on_step
    
    def validation_step(self,batch,batch_id) :
        imgs,targets = batch
        outputs = self.model(torch.stack(list(imgs), dim=0))
        # pdb.set_trace()
        loss_on_step = self.loss(outputs,torch.tensor(targets))
        #log on wandb
        if self.wandb_run:
            self.wandb_run.log({"val" : {"val" : loss_on_step}}, commit = True)
            for img, target, output in zip(imgs,targets,outputs):
                self.test_table.add_data('idx',wandb.Image(img),target,max(output),output[0],output[1],output[2],output[3],output[4])
        return
    
    def validation_epoch_end(self, outputs):
        self.wandb_run.log({'Val_table': self.test_table})
        self.test_table = self.test_table = wandb.Table(columns=self.columns)
        return 

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.lr)