# Dataset

import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy.lib.format
import pickle
import pydicom as dicom
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import glob
import random
import os
from utils import getPatiensIds



class CTCDataset(torch.utils.data.Dataset):
    def __init__(self, data_root,fromPkl,in_channels=1,patient_ids=[],train=True,seq_len=5,image_size=512,prefix_in = '.dcm',prefix_out = '.pgm',reverse=True):
        
        assert os.path.isfile(fromPkl), f'file{fromPkl} not exists'
        self.mean = None
        self.std=None
        self.reverse=reverse
        self.prefix_in = prefix_in
        self.prefix_out = prefix_out
        self.in_channels = in_channels
        self.patient_ids =patient_ids
        self.patientDict = {i:id  for i,id in  enumerate(patient_ids) }
        self.seq_len=seq_len
        
        
        self.get_mean_std_pkl(fromPkl)
        self.data_root =data_root
        self.dataDict,self.seq_list= self.getDataDict()
        self.train=train
        self.normalize_image = T.Normalize((self.mean),
                                            (self.std))
        self.image_size = image_size
        if image_size != 512:
            self.resize_image = T.transforms.Resize(image_size)
        else:
            self.resize_image=None

    
    # def get_seq_list(self):
    #     for key, value in self.dataDict.items():
    #         for 
    def getDataDict(self):
        dataDict = dict.fromkeys(self.patient_ids,None)
        seq_list=[]
        for key in   dataDict.keys():
            scans_list = []
            for x in ['prone', 'supine'] :          
                temp = glob.glob(f'{self.data_root}/{key}/*/{x}/dicom/*{self.prefix_in}')
                temp.sort(reverse= self.reverse)
                for i, _ in enumerate(temp[:-self.seq_len-1]):
                    seq_list.append(temp[i:i+self.seq_len+1])
                scans_list.append(temp)
            dataDict[key] = scans_list
        return dataDict, seq_list

             


        
    def getRandomSeq(self, patientScans):
        proneOrSupine = random.randint(0,1)
        patientScan =patientScans[proneOrSupine]
        numSlices= len(patientScan)
        randSeqStart = random.randint(1,numSlices-self.seq_len-1)
        inSeqPaths=patientScan[randSeqStart:randSeqStart+self.seq_len]
        outSeqPathes= [self.getLabelPath(x) for x in inSeqPaths  ]
        initial_mask_path = self.getLabelPath(patientScan[randSeqStart-1])
        return inSeqPaths,outSeqPathes,initial_mask_path
    
    
    def getSeq(self, inSeqPaths):
        outSeqPathes= [self.getLabelPath(x) for x in inSeqPaths  ]
        initial_mask_path = outSeqPathes[0]
        return inSeqPaths[1:],outSeqPathes[1:],initial_mask_path
    
    def getPatientSeq(self, patientScans,i):
        inSeqPaths =patientScans[i]
        outSeqPathes= [self.getLabelPath(x) for x in inSeqPaths  ]
        return inSeqPaths,outSeqPathes
    
    def getSeqOfImages(self,inSeqPaths,outSeqPaths,initial_mask_path=None):
        inSeqImgs=[]
        outSeqImgs=[]
        for inSeqPath,outSeqPath in zip(inSeqPaths,outSeqPaths):
            dcm = dicom.dcmread(inSeqPath)
            dcm_img = dcm.pixel_array
            dcm_img=np.tile(dcm_img, (self.in_channels,1,1))
            dcm_img = np.expand_dims(dcm_img,axis=0)
            dcm_label = numpy.array(Image.open(outSeqPath))
            dcm_label[dcm_label == 255] = 1  # binary 
            dcm_label = np.expand_dims(dcm_label,axis=0)
            dcm_label = np.expand_dims(dcm_label,axis=0)
            
            inSeqImgs.append(dcm_img)
            outSeqImgs.append(dcm_label)
        
        inSeqImgs = np.concatenate(inSeqImgs,axis=0)
        outSeqImgs = np.concatenate(outSeqImgs,axis=0)
        if initial_mask_path :
            initial_mask = numpy.array(Image.open(initial_mask_path))
            initial_mask[initial_mask == 255] = 1  # binary 
            initial_mask = np.expand_dims(initial_mask,axis=0)
        else:
            initial_mask = np.zeros_like(dcm_label).squeeze(0)
            
        return inSeqImgs,outSeqImgs,initial_mask


    def __getitem__(self, index):
        
        initial_mask_path = None
        # if self.train:
        seg_paths = self.seq_list[index]
        inSeqPaths,outSeqPaths, initial_mask_path= self.getSeq(seg_paths)
        # else:
        #     i = index//2
        #     j = index%2
        #     patiendId= self.patientDict[i]
        #     patient_scanns = self.dataDict[patiendId]
        #     inSeqPaths,outSeqPaths = self.getPatientSeq(patient_scanns,j)
        
        inSeqImgs,outSeqImgs,initMask = self.getSeqOfImages(inSeqPaths,outSeqPaths,initial_mask_path)
   
        
        outSeqImgs = torch.from_numpy(outSeqImgs)
        inSeqImgs = torch.from_numpy(inSeqImgs.astype('int32')).to(torch.float32)
        initMask = torch.from_numpy(initMask)

        if self.resize_image!=None :
            inSeqImgs = self.resize_image(inSeqImgs)
            outSeqImgs=self.resize_image(outSeqImgs)
            initMask = self.resize_image(initMask)
        
        inSeqImgs = self.normalize_image(inSeqImgs)

        if random.random() > 0.5 and self.train:
            angle = random.randint(-90, 90)
            inSeqImgs = TF.rotate(inSeqImgs, angle)
            outSeqImgs = TF.rotate(outSeqImgs, angle)
            initMask = TF.rotate(initMask, angle)
        
        # meta_inf={'dcm_meta':dcm_meta.to_json_dict(),'img_path':dcm_img_path,'label_path':dcm_label_path}

        meta_inf={'img_paths':inSeqPaths,'label_paths':outSeqPaths}

     
        return inSeqImgs.float(), outSeqImgs.float(),initMask.float(), meta_inf

    def getLabelPath(self,dcm_img_path):
        dcm_label_path=dcm_img_path.replace('/dicom/','/dicom/auto/segment.')
        dcm_label_path=dcm_label_path.replace(self.prefix_in,self.prefix_out)
        return dcm_label_path
    
    
    def set_train(self,train):
        self.trn=train
        
    def __len__(self):
        # if  self.train:
        return len(self.seq_list)
        # else:
        #     return len(self.dataDict)*2

    def get_no_of_channels(self):
        return len(self.__getitem__(0)[0])



    def cal_mean_std(self,save_path='./'):

        std =[]
        mean=[]
        for tem in self.data_path :

            dcm_img = dicom.dcmread(tem)
            dcm_img = dcm_img.pixel_array.astype(float)
            mean.append(np.mean(dcm_img))
            std.append(np.mean(dcm_img**2))
        self.mean = np.mean(mean)
        self.std = np.sqrt(np.mean(std)-self.mean**2)
        print('finished mean,std calc:', self.mean,self.std)
        self.normalize_image = T.Normalize((self.mean),(self.std))
        norm_dic = {'mean': self.mean,'std':self.std}
        with open(save_path+'ctcNormalization.pkl', 'wb') as handle:
            pickle.dump(norm_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.mean,self.std
    
    def get_mean_std(self):
        return self.mean,self.std
    
    def get_mean_std_pkl(self,pkl_path):
        with open(pkl_path, 'rb') as handle:
            normPkl = pickle.load(handle)
        self.mean= normPkl['mean']
        self.std=normPkl['std']
        print(normPkl)
        self.mean=(self.mean,)*self.in_channels
        self.std=(self.std,)*self.in_channels
        self.normalize_image = T.Normalize(self.mean,self.std)
        
    def visualize_samples(self, num_images=1):
        # print('get Samples:',num_images )
        samples = []
        rows = 2  # Number of rows in the grid
        cols = self.seq_len  # Number of columns in the grid (assuming 2 images per row)

        # Create a new figure
        fig, axes = plt.subplots(rows, cols, figsize=(30, 5))
        random_integer = random.randint(1, len(self)-1)
        img,label,initMask,_= self.__getitem__(random_integer)
        for i in range(img.shape[0]):
            
            
            axes[0,i].imshow(img[i].numpy().squeeze(),cmap='gray')
            axes[1,i].imshow(label[i].numpy().squeeze(),cmap='gray')
        samples.append((img,label))
        
        plt.tight_layout()
        plt.show()
        return samples

# root_path ="/media/hd2/Colon/Data_GT_Annotaion"
# fromPkl='/media/hd1/home/Shared/CTC_CVIP/Software/DeepLearning/config/ctcNormalization.pkl'

# patient_ids = getPatiensIds(root_path)
# # train_size = 0.7 
# # valid_size = 0.2
# # test_size = 1- train_size - valid_size
# ds = CTCDataset(root_path,fromPkl=fromPkl,patient_ids=patient_ids[0])
# ds.visualize_samples()


class CTCDataset2steps(CTCDataset):
    def __init__(self, **kwards):
        super.__init__(**kwards)
        assert self.seq_len%2 !=0 and self.seq_len>1, F'sequence length must be greater than 1 and odd got:{self.seq_len}'