import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import AdamW
import torch.nn as nn
from torchmetrics.functional import dice
import time
# from nnunet.utilities.nd_softmax import softmax_helper
import numpy as np

import yaml
# python train.py DeepLabV3+ tu-resnest269e
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

with open("./config/config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)

class ColonBoundaryWeightedDiceLoss(nn.Module):
    def __init__(self, boundary_weight=2.0):
        super(ColonBoundaryWeightedDiceLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.dice_loss = smp.losses.DiceLoss(mode="binary", smooth=1e-3) 
        # self.dice_loss = smp.losses.DiceLoss(mode="binary", smooth=config['model']['loss_smooth'])  # first time to try it


    def forward(self, x, y_, boundary_map):
        # x=torch.softmax(x, dim=0) # bad dice 17%
        x= torch.sigmoid(x) # tried and work 97%
        # x=nn.LeakyReLU(x, negative_slope=0.01, inplace=False)
        dice_loss = self.dice_loss(x, y_)  # Calculate the Dice loss
        boundary_loss = torch.mean(boundary_map * x)  # Weighted boundary loss
        loss = (self.boundary_weight)*dice_loss + (1-self.boundary_weight) * boundary_loss
        # loss = (1-self.boundary_weight)*dice_loss + self.boundary_weight * boundary_loss

        return loss
    

class ColonModule(pl.LightningModule):
    def __init__(self, config, segModel = None, pretrainedModel=None, in_channels=1):
        super().__init__()
        
        self.save_hyperparameters(ignore=["pretrainedModel"])
        self.config = config
        self.pretrainedModel=pretrainedModel
        if self.pretrainedModel !=None :
            self.pretrainedModel.freeze()
            in_channels+=1


        self.model = segModel(
            encoder_name=config["encoder_name"],
            encoder_weights=config["encoder_weights"],
            in_channels=config["in_channels"],
            classes=1,
            activation=None,
        )
        your_desired_boundary_weight= 0.547
        # self.loss_module = smp.losses.DiceLoss(mode="binary", smooth=config["loss_smooth"])
        self.loss_module = ColonBoundaryWeightedDiceLoss(boundary_weight=your_desired_boundary_weight) # set boundary_weight to 1.0. This means that the boundary will have the same importance as the rest of the image in the loss function
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
        
        # imgs, labels,_ = batch
        # imgs, labels, boundary_map = batch  # Make sure boundary_map is part of your batch data
        imgs, labels, _, boundary_map = batch 
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
        loss = self.loss_module(preds, labels, boundary_map)
        # loss = self.loss_module(preds, labels)
        # print(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=8)

        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # imgs, labels,_ = batch
        imgs, labels, _, boundary_map = batch 

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
        # loss = self.loss_module(preds, labels)
        loss = self.loss_module(preds, labels, boundary_map)  # Pass boundary_map to the loss function
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