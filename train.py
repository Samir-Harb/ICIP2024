import sys
import pprint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import torch
from torch.utils.data import DataLoader, Subset
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ctcdataset import CTCDataset
from seg_model import ColonModule
from custom_losses import WeightedDiceLoss
from pytorch_lightning.loggers import TensorBoardLogger
# python train.py DeepLabV3+ timm-regnetx_320      
# python train.py DeepLabV3+ tu-resnest269e
def main():
    # Create a TensorBoardLogger instance
    logger = TensorBoardLogger("logs/", name="exp1")

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

    assert len(sys.argv) > 1, 'error you have to pass at least one argument'
    segmentor = sys.argv[1]
    assert segmentor in seg_models
    config['model']['seg_model'] = segmentor
    encoder = sys.argv[2]
    print(sys.argv)
    # segmentor = 'Unet++'
    # encoder = 'timm-resnest269e'

    in_channels = config['model']['in_channels']
    ctcDataset = CTCDataset(data_path=config['data_path'], in_channels=in_channels, fromPkl=config['pkl_normalize'], train=False)
    train_size = 0.7
    valid_size = 0.2
    test_size = 1 - train_size - valid_size
    train_idx, valid_idx, test_idx = ctcDataset.split_train_valid_test([train_size, valid_size, test_size])

    print(len(train_idx))  # 7786

    train_ds = Subset(ctcDataset, train_idx)
    val_ds = Subset(ctcDataset, valid_idx)
    test_ds = Subset(ctcDataset, test_idx)
    # no_of_channels = 1   # this worked 
    no_of_channels= config['model']['in_channels']
 

    print(f'Starting Training Using {segmentor}')

    config['model']['encoder_name'] = encoder

    data_loader_train = DataLoader(
        train_ds,
        drop_last=True, 
        batch_size=config["train_bs"],
        shuffle=True,
        num_workers=config["workers"],
    )
    data_loader_validation = DataLoader(
        val_ds,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        monitor="val_dice",
        dirpath=config["output_dir"] + "/" + segmentor,
        mode="max",
        filename=segmentor + encoder + '-{val_dice:.4f}',
        save_last=True,
        save_top_k=1,
        verbose=1,
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=config["progress_bar_refresh_rate"]
    )

    # Update the early stopping configuration to monitor 'val_dice' instead of 'val_loss'
    # early_stop_callback = EarlyStopping(
    #     **config["early_stop"]
    # )
    # early_stop_callback = EarlyStopping(
    # monitor="val_dice",
    # **config["early_stop"]
    # )
    early_stop_callback = EarlyStopping(**config["early_stop"])

    pl.seed_everything(config['seed'])

    # Set up the trainer with the logger
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback],
        logger=logger,
        **config["trainer"]
    )

    config["model"]["scheduler"]["params"]["CosineAnnealingLR"]["T_max"] *= len(data_loader_train) / config["trainer"]["devices"]

    segModel = seg_models[segmentor]
    pretrainedModel = None

    model = ColonModule(config=config["model"], segModel=segModel, pretrainedModel=pretrainedModel, in_channels=no_of_channels)

    pprint.pprint(config)
    trainer.fit(model, data_loader_train, data_loader_validation)

    # Access the log directory and print it
    log_dir = trainer.log_dir
    print(f"Log directory: {log_dir}")

if __name__ == '__main__':
    main()
