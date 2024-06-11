import torch
from typing import Any, Callable, List, Optional, Type, Union
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.optim import AdamW
import torch.nn as nn
from torchmetrics.functional import dice
import time
from torch import Tensor
from convLSTM import ConvLSTM,ConvLSTMCell


seg_models = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
}

class ColonModule(pl.LightningModule):
    def __init__(self, config, segModel = None, pretrainedModel=None, in_channels=1):
        super().__init__()
        
        self.save_hyperparameters(ignore=["pretrainedModel"])
        print(config)
        self.config = config
        self.pretrainedModel=pretrainedModel
        if self.pretrainedModel !=None :
            self.pretrainedModel.freeze()
            in_channels+=1


        self.model = segModel(
            encoder_name=config["encoder_name"],
            # encoder_weights=config["encoder_weights"],
            # in_channels=config["in_channels"],
            in_channels=1,

            classes=1,
            activation=None,
        )

        self.loss_module = smp.losses.DiceLoss(mode="binary", smooth=config["loss_smooth"])
        self.val_step_outputs = []
        self.val_step_labels = []


    def forward(self, batch):
        imgs = batch
        
        if self.pretrainedModel !=None:
            self.pretrainedModel.eval()
            with torch.no_grad():
                initialMask = self.pretrainedModel(imgs)
                initialMask = torch.sigmoid(initialMask)
            
            imgMask = torch.cat((imgs, initialMask), 1)    
            preds = self.model(imgMask)
        else:
           preds = self.model(imgs) 
        # et = time.time()
        # print(f'time for forward path: {et-st}')
        return preds

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.config["optimizer_params"])

        if self.config["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                **self.config["scheduler"]["params"]["CosineAnnealingLR"],
            )
            lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        elif self.config["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                **self.config["scheduler"]["params"]["ReduceLROnPlateau"],
            )
            lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    
    def training_step(self, batch, batch_idx):
        
        imgs, labels,_ = batch
        # print(imgs.shape)
        
        if self.pretrainedModel !=None:
            self.pretrainedModel.eval()
            with torch.no_grad():
                initialMask = self.pretrainedModel(imgs)
                initialMask = torch.sigmoid(initialMask)
            imgMask = torch.cat((imgs, initialMask), 1)
            preds = self.model(imgMask)
        else:
           preds = self.model(imgs) 
        
        if self.config["image_size"] != 512:
            preds = torch.nn.functional.interpolate(preds, size=512, mode='bilinear')
        loss = self.loss_module(preds, labels)
        # print(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=8)

        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels,_ = batch
        # print((imgs.shape))
        if self.pretrainedModel !=None:
            initialMask = self.pretrainedModel(imgs)
            initialMask = torch.sigmoid(initialMask)
            imgMask = torch.cat((imgs, initialMask), 1)
            preds = self.model(imgMask)
        else:
           preds = self.model(imgs) 
        
        if self.config["image_size"] != 512:
            preds = torch.nn.functional.interpolate(preds, size=512, mode='bilinear')
        loss = self.loss_module(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds.cpu())
        self.val_step_labels.append(labels.cpu())

    def on_validation_epoch_end(self):
        print(len(self.val_step_outputs))
        all_preds = torch.cat(self.val_step_outputs).float()
        all_labels = torch.cat(self.val_step_labels)

        all_preds = torch.sigmoid(all_preds)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        # print(np.unique(all_labels.long().to('cpu').numpy()))
        val_dice = dice(all_preds, all_labels.long())
        self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=True)
        # print("val_dice", val_dice)
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)

class ColonModuleLSTM(pl.LightningModule):
    def __init__(self, config, segModel = None):
        super().__init__()
        
        self.save_hyperparameters()
        self.config = config
        # model_param = config['model']
        convLSTM_hidden_chs = config['convLSTM_hidden_chs']
        # self.convLSTM = ConvLSTM(input_channels=model_param['in_channels'], 
        #                          hidden_channels=model_param['convLSTM_hidden_chs'], 
        #                          kernel_size=model_param['convLSTM_kSize'])
        self.convLSTMCells = self.biuld_convLSTM_cells(config)
        
        
        self.lstmOutConv = nn.Conv2d(convLSTM_hidden_chs[-1], config['in_channels'],1)
        self.model = segModel(
            encoder_name=config["encoder_name"],
            encoder_weights=config["encoder_weights"],
            in_channels=config['in_channels']*2,
            classes=1,
            activation=None,
        )

        self.loss_module = smp.losses.DiceLoss(mode="binary", smooth=config["loss_smooth"])
        self.val_step_outputs = []
        self.val_step_labels = []
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.config["optimizer_params"])

        if self.config["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                **self.config["scheduler"]["params"]["CosineAnnealingLR"],
            )
            lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        elif self.config["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                **self.config["scheduler"]["params"]["ReduceLROnPlateau"],
            )
            lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def biuld_convLSTM_cells(self,model_param):
        self.input_channels = [model_param['in_channels']] + model_param['convLSTM_hidden_chs']
        self.hidden_channels = model_param['convLSTM_hidden_chs']
        self.kernel_size = model_param['convLSTM_kSize']
        self.num_layers = len(self.hidden_channels)
        convLSTMCells = nn.ModuleList()
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            convLSTMCells.append(cell)
        return convLSTMCells     
    
    
    def forward(self, imgs, initMask):
        pred_array=[initMask]
            
        # hidden_state=self.convLSTM(initMask)
        internal_states = []
        num_steps = imgs.shape[1]
        for step in range(num_steps):
            x = pred_array[step]
            for i, cell in enumerate(self.convLSTMCells):
                # all cells are initialized in the first step
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = cell.init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_states.append((h, c))

                # do forward
                (h, c) = internal_states[i]
                x, new_c = cell(x, h, c)
                internal_states[i] = (x, new_c)
            x = self.lstmOutConv(x)
            
            
            
            x = torch.cat([imgs[:,step,...],x],dim=1)

            preds = self.model(x)
            pred_array.append(preds)
        
        preds = torch.cat(pred_array[1:],dim=1) 
        # et = time.time()
        # print(f'time for forward path: {et-st}')
        return preds

    
    
    def training_step(self, batch, batch_idx):
        
        imgs, labels,initMask,_ = batch
        # print(imgs.shape)
        
        
        preds = self(imgs,initMask) 
        # print(preds.shape)
        
        if self.config["image_size"] != 512:
            preds = torch.nn.functional.interpolate(preds, size=512, mode='bilinear')
        
        # print(preds.shape,labels)
        loss = self.loss_module(preds, labels)
        # print(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=8)

        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels,initMask,_ = batch
        # print((imgs.shape))
        preds = self(imgs,initMask) 
        # print('#####################',preds.shape)
        if self.config["image_size"] != 512:
            preds = torch.nn.functional.interpolate(preds, size=512, mode='bilinear')
        loss = self.loss_module(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds.cpu().squeeze())
        self.val_step_labels.append(labels.cpu().squeeze())

    def on_validation_epoch_end(self):
        # print(len(self.val_step_outputs))
        all_preds = torch.cat(self.val_step_outputs).float()
        all_labels = torch.cat(self.val_step_labels)

        all_preds = torch.sigmoid(all_preds)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        # print(np.unique(all_labels.long().to('cpu').numpy()))
        val_dice = dice(all_preds, all_labels.long())
        self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=True)
        # print("val_dice", val_dice)
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)


class ColonModule2Steps(pl.LightningModule):
    seg_models = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
    }
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters(ignore=["pretrainedModel"])
        self.config=config
        model1_config = config['model1']
        model2_config = config['model2']
        self.model1_config = model1_config
        self.model2_config = model2_config
        segModel1 = model1_config ['seg_model']
        segModel2 = model2_config ['seg_model']
        self.model1 = self.seg_models[segModel1](
            encoder_name=model1_config["encoder_name"],
            encoder_weights=model1_config["encoder_weights"],
            in_channels=model1_config["in_channels"],
            classes=1,
            activation=None,
        )
        self.model2 = self.seg_models[segModel2](
            encoder_name=model2_config["encoder_name"],
            encoder_weights=model2_config["encoder_weights"],
            in_channels=model2_config["in_channels"],
            classes=1,
            activation=None,
        )

        self.loss_module1 = smp.losses.DiceLoss(mode="binary", smooth=model1_config["loss_smooth"])
        self.loss_module2 = smp.losses.DiceLoss(mode="binary", smooth=model2_config["loss_smooth"])
        
        self.val_step_outputs = []
        self.val_step_labels = []
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.model1.parameters())
        optimizer2 = torch.optim.Adam(self.model2.parameters())
        optimizers = [optimizer1,optimizer2] 
        
        # Apply lr scheduler per step
        lr_scheduler1  = OneCycleLR(optimizer1, 
                                    max_lr=self.model1_config["scheduler"]["params"]["max_lr"], 
                                    total_steps =self.model1_config["scheduler"]["params"]['max_iters'])
        lr_scheduler2  = OneCycleLR(optimizer2, 
                                    max_lr=self.model2_config["scheduler"]["params"]["max_lr"], 
                                    total_steps =self.model2_config["scheduler"]["params"]['max_iters'])
        schedulers =[lr_scheduler1,lr_scheduler2] 
        
        return  optimizers, schedulers
    
    

    
    def forward(self, batch):
        imgs = batch
        B, S, C, W, H= batch.shape

        imgs = imgs.flatten(0,1)

        preds_1 = self.model1(imgs)
        preds_1 = torch.reshape(preds_1,(B,S,1,W,H))
        preds_2 = self.model2(preds_1.squeeze().detach())
        # et = time.time()
        # print(f'time for forward path: {et-st}')
        return preds_1, preds_2

    
       

    
    def training_step(self, batch, batch_idx):
        imgs, labels,_,_ = batch

        M=imgs.shape[1]//2
        
        preds1, preds2 = self(imgs)
        
        if self.model1_config["image_size"] != 512:
            preds1 = torch.nn.functional.interpolate(preds1, size=512, mode='bilinear')
            preds2 = torch.nn.functional.interpolate(preds2, size=512, mode='bilinear')
        loss1 = self.loss_module1(preds1, labels)
        loss2 = self.loss_module2(preds2, labels[...,M,:,:,:])
        
        opt1, opt2 = self.optimizers()
        sch1, sch2 = self.lr_schedulers()
        opt1.zero_grad()
        self.manual_backward(loss1)
        opt1.step()
        opt2.zero_grad()
        self.manual_backward(loss2)
        opt2.step()
        sch1.step()
        sch2.step()
        # print(loss)
        self.log_dict({"loss1": loss1, "loss2": loss2}, on_step=True, on_epoch=True, prog_bar=True)


        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return 

    def validation_step(self, batch, batch_idx):
        imgs, labels,_,_ = batch

        M=imgs.shape[1]//2
        
        preds1, preds2 = self(imgs)
        
        if self.model1_config["image_size"] != 512:
            preds1 = torch.nn.functional.interpolate(preds1, size=512, mode='bilinear')
            preds2 = torch.nn.functional.interpolate(preds2, size=512, mode='bilinear')
        loss1 = self.loss_module1(preds1, labels)
        loss2 = self.loss_module2(preds2, labels[...,M,:,:,:])
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"val_loss1": loss1, "val_loss2": loss2}, on_step=True, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds2.cpu())
        self.val_step_labels.append(labels[...,M,:,:,:].cpu())

    def on_validation_epoch_end(self):
        print(len(self.val_step_outputs))
        all_preds = torch.cat(self.val_step_outputs).float()
        all_labels = torch.cat(self.val_step_labels)

        all_preds = torch.sigmoid(all_preds)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        # print(np.unique(all_labels.long().to('cpu').numpy()))
        val_dice = dice(all_preds, all_labels.long())
        self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=True)
        # print("val_dice", val_dice)
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)


class ColonModule2Steps(pl.LightningModule):
    seg_models = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
    }
    def __init__(self, config,model1=None,model2= None):
        super().__init__()
        
        self.save_hyperparameters(ignore=["model1","model2"])
        self.config=config
        model1_config = config['model1']
        model2_config = config['model2']
        self.model1_config = model1_config
        self.model2_config = model2_config
        segModel1 = model1_config ['seg_model']
        segModel2 = model2_config ['seg_model']
        self.model1Freezed=False
        if model1:
            self.model1Freezed=True
            self.model1 = model1
            self.model1.freeze()
        else:
            self.model1 = self.seg_models[segModel1](
                encoder_name=model1_config["encoder_name"],
                encoder_weights=model1_config["encoder_weights"],
                in_channels=model1_config["in_channels"],
                classes=1,
                activation=None,
            )
        
        self.segmoid = nn.Sigmoid()
        if model2:
            self.model2 = model2
        else:

            self.model2 = self.seg_models[segModel2](
                encoder_name=model2_config["encoder_name"],
                encoder_weights=model2_config["encoder_weights"],
                in_channels=model2_config["in_channels"],
                classes=1,
                activation=None,
            )

        self.loss_module1 = smp.losses.DiceLoss(mode="binary", smooth=model1_config["loss_smooth"])
        self.loss_module2 = smp.losses.DiceLoss(mode="binary", smooth=model2_config["loss_smooth"])
        
        self.val_step_outputs = []
        self.val_step_labels = []
        self.automatic_optimization = False


    def configure_optimizers(self):
        optimizer1 = AdamW(self.model1.parameters(), **self.model1_config["optimizer_params"])
        optimizer2 = AdamW(self.model2.parameters(), **self.model2_config["optimizer_params"])
        optimizers = [optimizer1,optimizer2] 

        scheduler1 = CosineAnnealingLR(
            optimizer1,
            **self.model1_config["scheduler"]["params"]["CosineAnnealingLR"],
        )
        scheduler2 = CosineAnnealingLR(
            optimizer2,
            **self.model2_config["scheduler"]["params"]["CosineAnnealingLR"],
        )
        schedulers =[scheduler1,scheduler2] 
        
        
        return optimizers, schedulers

    

    
    def forward(self, batch):
        imgs = batch
        B, S, C, W, H= batch.shape
        M = S//2

        imgs = imgs.flatten(0,1)

        preds_1 = self.model1(imgs)
        preds_1 = torch.reshape(preds_1,(B,S,1,W,H))
        imgs = torch.reshape(imgs,(B,S,C,W,H))
        atten_map = self.segmoid(preds_1.sum(dim=1))
        attended_im= imgs[:,M,:,:,:]*atten_map
        preds_2 = self.model2(attended_im.detach())
        # et = time.time()
        # print(f'time for forward path: {et-st}')
        return preds_1, preds_2

    
       

    
    def training_step(self, batch, batch_idx):
        imgs, labels,_,_ = batch

        M=imgs.shape[1]//2
        
        preds1, preds2 = self(imgs)
        
        if self.model1_config["image_size"] != 512:
            preds1 = torch.nn.functional.interpolate(preds1, size=512, mode='bilinear')
            preds2 = torch.nn.functional.interpolate(preds2, size=512, mode='bilinear')
        opt1, opt2 = self.optimizers()
        sch1, sch2 = self.lr_schedulers()
        if not self.model1Freezed:
            loss1 = self.loss_module1(preds1, labels)
            opt1.zero_grad()
            self.manual_backward(loss1)
            opt1.step()
            sch1.step()
        
        
        loss2 = self.loss_module2(preds2, labels[...,M,:,:,:])
        
        
        opt2.zero_grad()
        self.manual_backward(loss2)
        opt2.step()
        sch2.step()
        # print(loss)
        if self.model1Freezed:
            self.log_dict({ "loss2": loss2}, on_step=True, on_epoch=True, prog_bar=True)
        
        else:
            self.log_dict({"loss1": loss1, "loss2": loss2}, on_step=True, on_epoch=True, prog_bar=True)


        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return 

    def validation_step(self, batch, batch_idx):
        imgs, labels,_,_ = batch

        M=imgs.shape[1]//2
        
        preds1, preds2 = self(imgs)
        
        if self.model1_config["image_size"] != 512:
            preds1 = torch.nn.functional.interpolate(preds1, size=512, mode='bilinear')
            preds2 = torch.nn.functional.interpolate(preds2, size=512, mode='bilinear')
        loss1 = self.loss_module1(preds1, labels)
        loss2 = self.loss_module2(preds2, labels[...,M,:,:,:])
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"val_loss1": loss1, "val_loss2": loss2}, on_step=True, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds2.cpu())
        self.val_step_labels.append(labels[...,M,:,:,:].cpu())

    def on_validation_epoch_end(self):
        print(len(self.val_step_outputs))
        all_preds = torch.cat(self.val_step_outputs).float()
        all_labels = torch.cat(self.val_step_labels)

        all_preds = torch.sigmoid(all_preds)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        # print(np.unique(all_labels.long().to('cpu').numpy()))
        val_dice = dice(all_preds, all_labels.long())
        self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=True)
        # print("val_dice", val_dice)
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)
