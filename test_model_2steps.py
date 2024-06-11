


import sys
import pprint
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import torch
from torch.utils.data import DataLoader,Subset

import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from seq_ctcdataset import CTCDataset
from seg_model import ColonModule,ColonModule2Steps
from utils import getPatiensIds
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


def testModel(model,test_dl,outpath='',device='cpu'):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    model=model.to(device)
    model.eval()
    model.zero_grad()
    count=0
    out_map=[]
    for i, batch in tqdm(enumerate(test_dl)):
        images, labels,_,meta_ifo = batch

        M=images.shape[1]//2

        images = images.to(device)
        with torch.no_grad():
            _, predicted_mask = model(images)
        predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=512, mode='bilinear')
        predicted_mask = torch.sigmoid(predicted_mask).cpu().detach().numpy()
        predicted_mask_with_threshold = np.zeros((images.shape[0], 512, 512))
        predicted_mask_with_threshold[predicted_mask[:, 0, :, :] > 0.3] = 255
        for img_num in range(0, images.shape[0]):
            pre_mask = predicted_mask_with_threshold[img_num, :, :].astype('uint8').squeeze()
            gt_mask=labels[img_num,M,:,:,:].cpu().detach().numpy()*255
            gt_mask = gt_mask.astype('uint8').squeeze()
            in_image=images[img_num,M,:,:,:].cpu().detach().numpy()
            in_image = (in_image-np.min(in_image))/(np.max(in_image)-np.min(in_image))*255
            in_image=in_image.astype('uint8')
            test_sample =np.concatenate([in_image[0,:,:],gt_mask,pre_mask],axis=1)
            im = Image.fromarray(test_sample)
            im.save(f"{outpath}test_sample_{count}.png")
            out_map.append([str(count),meta_ifo['img_paths'][M][img_num],meta_ifo['label_paths'][M][img_num]])
            count+=1
    out_map=np.array(out_map)
    df = pd.DataFrame(out_map, columns =['index', 'img_path', 'label_path']) 
    df.to_csv(f'{outpath}out_map.csv')


def main():
    
    
    
    with open("/media/hd1/home/Shared/CTC_CVIP/Software/DeepLearning/config/config_2steps.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)
    
    segmentor1= config['model1']['seg_model']
    segmentor2= config['model2']['seg_model']
    encoder1= config['model1']['encoder_name']
    encoder2= config['model2']['encoder_name']
    

    
    ids_train, ids_valid, ids_test = getPatiensIds(config['data_path'])
    # train_ds = CTCDataset(config['data_path'],fromPkl=config['pkl_normalize'],patient_ids=ids_train,seq_len=config['seq_len'])
    # valid_ds = CTCDataset(config['data_path'],fromPkl=config['pkl_normalize'],patient_ids=ids_valid,seq_len=config['seq_len'])
    test_ds  = CTCDataset(config['data_path'],fromPkl=config['pkl_normalize'],patient_ids=ids_test, seq_len=config['seq_len'],train=False)
    
    print('train:',len(ids_train),'valid:',len(ids_valid),'test:',len(ids_test))
 
    print(f'Starting Training Using, {segmentor1}, {segmentor2}, {encoder1}, {encoder2}')
    
    

    test_dl = DataLoader(
        test_ds,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )

    model_path1 = config['model1']['model_path']
    model1   = ColonModule.load_from_checkpoint(model_path1,config=config['model1'],map_location='cpu')


    # model = ColonModule2Steps(config = config,model1=model1)
    model_path2  = config['model2']['model_path']
    model = ColonModule2Steps.load_from_checkpoint(model_path2,model1=model1,map_location='cpu')

 
    
    # model1   = ColonModule.load_from_checkpoint(model_path,config=config['model1'],map_location='cpu')


    # model = ColonModule2Steps(config = config,model1=model1)
    # model = ColonModule2Steps.load_from_checkpoint(model_path,map_location='cpu')
    # model = ColonModuleLSTM.load_from_checkpoint(config['saved_model_path'])
    pprint.pprint(config)
    # trainer.fit(model, data_loader_train, data_loader_validation)
    experiment = 'Unet2-resnest26d'
    outpath =f"{config['base_dir']}{config['test_save_dir']}{experiment}/"
    testModel(model,test_dl,outpath=outpath,device='cpu')
if __name__=='__main__':
    main()

        

