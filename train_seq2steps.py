
import sys
import pprint

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


def main():
    
    
    
    with open("/media/hd1/home/Shared/CTC_CVIP/Software/DeepLearning/config/config_2steps.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)
    
    segmentor1= config['model1']['seg_model']
    segmentor2= config['model2']['seg_model']
    encoder1= config['model1']['encoder_name']
    encoder2= config['model2']['encoder_name']
    


    in_channels=config['model1']['in_channels']
    
    ids_train, ids_valid, ids_test = getPatiensIds(config['data_path'])
    train_ds = CTCDataset(config['data_path'],fromPkl=config['pkl_normalize'],patient_ids=ids_train,seq_len=config['seq_len'])
    valid_ds = CTCDataset(config['data_path'],fromPkl=config['pkl_normalize'],patient_ids=ids_valid,seq_len=config['seq_len'],train=False)
    # test_ds  = CTCDataset(config['data_path'],fromPkl=config['pkl_normalize'],patient_ids=ids_test, seq_len=config['seq_len'],train=False)
    
    print('train:',len(ids_train),'valid:',len(ids_valid),'test:',len(ids_test))

    no_of_channels = 1
 
    print(f'Starting Training Using, {segmentor1}, {segmentor2}, {encoder1}, {encoder2}')
    
    

    
    
    # config['model']['encoder_name'] = encoder
    # if segmentor in ['PAN','DeepLabV3','DeepLabV3+']:
    #     config['model']['encoder_name'] = 'tu-resnest269e'
    #     config['train_bs'] = 32
    #     config['valid_bs'] = 32

    data_loader_train = DataLoader(
        train_ds,
        drop_last=False,
        batch_size=config["train_bs"],
        shuffle=True,
        num_workers=config["workers"],
    )
    data_loader_validation = DataLoader(
        valid_ds,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        monitor="val_dice",
        dirpath=config["output_dir"]+"/"+segmentor1+"-"+segmentor2,
        mode="max",
        filename=encoder1+'-'+encoder2+'-{val_dice:.4f}',
        save_last=True,
        save_top_k=1,
        verbose=1,
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=config["progress_bar_refresh_rate"]
    )

    early_stop_callback = EarlyStopping(**config["early_stop"])
    pl.seed_everything(config['seed'])
    trainer = pl.Trainer(
        # check_val_every_n_epoch = None,
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback],
        **config["trainer"],
    )

    # config["model1"]["scheduler"]["params"]["CosineAnnealingLR"]["T_max"] *= len(data_loader_train)/config["trainer"]["devices"]
    # config["model2"]["scheduler"]["params"]["CosineAnnealingLR"]["T_max"] *= len(data_loader_train)/config["trainer"]["devices"]
    config['model1']["scheduler"]["params"]["max_iters"]= trainer.max_epochs * len(data_loader_train)
    config['model2']["scheduler"]["params"]["max_iters"]= trainer.max_epochs * len(data_loader_train)
    # config['model2']['in_channels']=config['seq_len']
    model_path = config['model1']['model_path']
    model1   = ColonModule.load_from_checkpoint(model_path,config=config['model1'])


    model = ColonModule2Steps(config = config,model1=model1)
    # model = ColonModuleLSTM.load_from_checkpoint(config['saved_model_path'])
    pprint.pprint(config)
    trainer.fit(model, data_loader_train, data_loader_validation)
if __name__=='__main__':
    main()

        