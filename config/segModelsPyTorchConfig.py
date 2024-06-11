# Save the following codes in example_config.py
# Almost copied from the above example, with some commas removed
inChannels=3
data_path = '/media/hd2/Colon/Data_GT_Annotaion'
fromPkl = "/media/hd1/home/Shared/CTC_CVIP/Software/DeepLearning/config/ctcNormalization.pkl"

splitRatio=[0.7,0.2,0.1]
work_dir = './work_dir'
max_epochs=20

model = dict(type='MMSegModelsPyTorch',
    modelName='Unet++', 
    encoderName='tu-resnest269e',
    encoderWeights='imagenet',
    inChannels=inChannels,
    loss_smooth= 1.0)



randomness=dict(seed=400,diff_rank_seed=False)
trainDS=dict(type='MMCTCDataset',
        data_path=data_path,
        fromPkl=fromPkl,
        inChannels=inChannels,
        train=True,
        splitRatio=splitRatio)




train_dataloader = dict(
    dataset=trainDS,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=8,
    # pin_memory=True,
    num_workers=1)

train_cfg = dict(
    by_epoch=True,
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=1)

optimizer=dict(type='AdamW', lr=1e-4,weight_decay=0.03)

optim_wrapper=dict(type='AmpOptimWrapper', optimizer=optimizer)

param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, convert_to_iter_based=True)


validDS=dict(type='MMCTCDataset',
        data_path=data_path,
        fromPkl=fromPkl,
        train=False,
        inChannels=inChannels,
        splitRatio=splitRatio)
val_dataloader = dict(
    dataset=validDS,
    sampler=dict(type='DefaultSampler',shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=16,
    pin_memory=True,
    num_workers=1)
val_cfg = dict()
val_evaluator = dict(type='DiceScore')

default_hooks=dict(
        checkpoint=dict(type='CheckpointHook', interval=1),
        logger=dict(type='LoggerHook',interval=100),)
custom_hooks = [dict(type='SegVisHook', data_root=data_path)]
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
# load_from='/media/hd1/home/Shared/CTC_CVIP/work_dir/epoch_10.pth'
log_level = 'INFO'
load_from = None
# resume = True

