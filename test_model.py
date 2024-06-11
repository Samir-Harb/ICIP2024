
import sys
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import torch
from torch.utils.data import DataLoader,Subset
import pytorch_lightning as pl
import yaml
import numpy as np
import pandas as pd
from PIL import Image


from ctcdataset import CTCDataset
from seg_model import ColonModule


def main():
    # assert len(sys.argv) > 1, 'error you have to pass at least two arguements'
    # experiment = sys.argv[1]

    # experiment= 'Unettu-resnest269e-val_dice=0.9786.ckpt'
    # experiment= 'Unet++_timm-resnest269e-fold0-val_dice=0.9895.ckpt'
    # experiment= 'Unet++_timm-resnest269e-fold0-val_dice=0.9895.ckpt_savepredonly'
    # experiment= 'Unet++timm-resnest269e-val_dice=0.9796.ckpt'
    experiment= 'Unet++timm-resnest269e-val_dice=0.9796.ckpt_savepredonly'
    with open("./config/config.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)

    in_channels=config['model']['in_channels']
    ctcDataset= CTCDataset(data_path=config['data_path'],in_channels=in_channels,fromPkl=config['pkl_normalize'],train=False)
    
    train_size = 0.7 
    valid_size = 0.2
    test_size = 1- train_size - valid_size
    _, _, test_idx = ctcDataset.split_train_valid_test([train_size,valid_size,test_size])
    # test_ds = Subset(ctcDataset,test_idx)
    test_idx_subset = test_idx[:429]  # Use a subset of the test indices
    test_ds = Subset(ctcDataset,test_idx_subset) # pick a specific volume whose indices are already known
 
    test_dl= DataLoader(
        test_ds,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )
    pl.seed_everything(config['seed'])
    trainer = pl.Trainer(accelerator="cpu",
        **config["trainer"],
    )

    # Print size per image in the test dataset
    for i, data in enumerate(test_dl):
        images, _, _, _ = data
        print(f"Image {i + 1} size: {images.size()}")
        break

    model = ColonModule.load_from_checkpoint(config['saved_model_path'],map_location='cpu')
    # model=model.to('cpu')
    # trainer.validate(model, test_dl)
    print('model is loaded --> start testing')
    outpath =f"{config['base_dir']}{config['test_save_dir']}{experiment}/"
    # testModel_save_ct_gt_pred(model,test_dl,outpath=outpath,device='cpu')
    testModel_save_pred_mask(model,test_dl,outpath=outpath,device='cpu')
    # trainer.fit(model, train_dl, valid_dl)
def testModel_save_ct_gt_pred(model,test_dl,outpath='',device='cpu'):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    model=model.to(device)
    model.eval()
    model.zero_grad()
    count=0
    out_map=[]
    for i, data in enumerate(test_dl):

        images, labels,meta_ifo,_ = data
        images = images.to(device)
        with torch.no_grad():
            predicted_mask = model.forward(images)
        predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=512, mode='bilinear')
        predicted_mask = torch.sigmoid(predicted_mask).cpu().detach().numpy()
        predicted_mask_with_threshold = np.zeros((images.shape[0], 512, 512))
        predicted_mask_with_threshold[predicted_mask[:, 0, :, :] > 0.3] = 255
        for img_num in range(0, images.shape[0]):
            pre_mask = predicted_mask_with_threshold[img_num, :, :].astype('uint8').squeeze()
            gt_mask=labels[img_num,:,:,:].cpu().detach().numpy()*255
            gt_mask = gt_mask.astype('uint8').squeeze()
            in_image=images[img_num,:,:,:].cpu().detach().numpy()
            in_image = (in_image-np.min(in_image))/(np.max(in_image)-np.min(in_image))*255
            in_image=in_image.astype('uint8')
            test_sample =np.concatenate([in_image[0,:,:],gt_mask,pre_mask],axis=1)
            im = Image.fromarray(test_sample)
            im.save(outpath+"test_sample_"+"{:03d}".format(count)+".png")
            out_map.append([str(count),meta_ifo['img_path'][img_num],meta_ifo['label_path'][img_num]])
            count+=1
    out_map=np.array(out_map)
    df = pd.DataFrame(out_map, columns =['index', 'img_path', 'label_path']) 
    df.to_csv(f'{outpath}out_map.csv')

def testModel_save_pred_mask(model, test_dl, outpath='', device='cpu'):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    model.eval()  # Sets the model to evaluation mode (model.eval())
    model.zero_grad()
    count = 0
    out_map = []

    for i, data in enumerate(test_dl):
        images, _, meta_ifo, _ = data
        print(images.shape)
        images = images.to(device)

        with torch.no_grad():
            predicted_mask = model.forward(images)

        # Process the predicted mask
        predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=512, mode='bilinear')
        predicted_mask = torch.sigmoid(predicted_mask).cpu().detach().numpy()
        predicted_mask_with_threshold = np.zeros((images.shape[0], 512, 512))
        predicted_mask_with_threshold[predicted_mask[:, 0, :, :] > 0.3] = 255

        for img_num in range(0, images.shape[0]):
            pre_mask = predicted_mask_with_threshold[img_num, :, :].astype('uint8').squeeze()

            # Save only the predicted mask
            im = Image.fromarray(pre_mask)
            # im.save(f"{outpath}test_sample_{count}.png")
            im.save(outpath+"test_sample_"+"{:03d}".format(count)+".png")
            out_map.append([str(count), meta_ifo['img_path'][img_num], meta_ifo['label_path'][img_num]])
            count += 1

    out_map = np.array(out_map)
    df = pd.DataFrame(out_map, columns=['index', 'img_path', 'label_path'])
    df.to_csv(f'{outpath}out_map.csv')
    

if __name__=='__main__':
    main()

        