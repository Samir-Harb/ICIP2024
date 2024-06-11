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

import numpy as np
from scipy.ndimage import distance_transform_edt

import glob
import random
import os

# Dataset
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
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

import albumentations as A
from torchvision.transforms import v2
import cv2


class CTCDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,fromPkl,in_channels=1,train=False,image_size=512):
        
        assert os.path.isfile(fromPkl), f'file{fromPkl} not exists'
        self.mean = None
        self.std=None
        self.in_channels = in_channels
        
        
        self.get_mean_std_pkl(fromPkl)
        

   
        self.data_path = glob.glob(data_path+'/**/dicom/*.dcm',recursive=True) or ...
        glob.glob(data_path + '/**/dicom/*.DCM', recursive=True)
        
        self.data_path.sort(reverse=False)
        data_path[-1]
        self.patients_ids = list(set([ x.split('/')[-5] for x in self.data_path ]))
        self.patients_ids.sort()
        self.trn = train
        self.normalize_image = T.Normalize((self.mean),
                                            (self.std))
        self.image_size = image_size
        if image_size != 512:
            self.resize_image = T.transforms.Resize(image_size)
        else:
            self.resize_image=None
        
                # Define deformable augmentations
       
        self.deform_aug = A.Compose([
            # A.RandomSizedCrop(min_max_height=(450, 512), height=512, width=512, p=1),
            # A.VerticalFlip(p=1),              
            # A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, num_steps=130, distort_limit=0.9),
            #  A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, num_steps=105, distort_limit=0.8),   dice=0.9818,  p=1
            # A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, num_steps=130, distort_limit=0.8), dice=0.98188,  p=1
            #A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, num_steps=30, distort_limit=0.8),  dice=0.98188,  p=1
            # A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, num_steps=100, distort_limit=0.8), dice=0.98288,  p=1
            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5),

            # Add more deformable augmentations as needed
            ], p=1)  # You can adjust the probability 'p' as needed
         
        self.transforms = Compose([
            Resize(width = 512, height = 512),
            # Normalize(mean=841.7832213411665, std=1508.985814857652, p=1), #old parameters without adjusting visualization
            # Normalize(mean= -198.32743039563474, std= 352.6458884130853, p=1), #updated parameters with adjusting visualization
            Normalize(mean= self.mean, std= self.std, p=1), #updated parameters with adjusting visualization
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ToTensorV2(),
        ])

    def __getitem__(self, index):
        
        dcm_img_path = self.data_path[index]
        # print(dcm_img_path)
        
        dcm = dicom.dcmread(dcm_img_path)
        
        # '''Samir  to adjust visualization quality'''  
        # dcm_meta=dcm.file_meta
        # image = dcm.pixel_array.astype('float')
        # image = image*dcm.RescaleSlope + dcm.RescaleIntercept
        # window_min = dcm.WindowCenter-dcm.WindowWidth/2
        # window_max = dcm.WindowCenter+dcm.WindowWidth/2
        # dcm_img = np.clip(image,window_min,window_max)        
        # '''Samir  to adjust visualization quality''' 
        
        dcm_meta=dcm.file_meta
        dcm_img = dcm.pixel_array
        dcm_img=np.tile(dcm_img, (self.in_channels,1,1))
        dcm_label_path = self.getLabelPath(dcm_img_path,dcm_meta)
        dcm_label = numpy.array(Image.open(dcm_label_path))
        dcm_label[dcm_label == 255] = 1  # binary 
        dcm_label = np.expand_dims(dcm_label,axis=0)
        
        # Calculate the boundary_map
        boundary_map = self.compute_boundary_map(dcm_label)

        dcm_label = torch.from_numpy(dcm_label)
        dcm_img = torch.from_numpy(dcm_img.astype('int32')).to(torch.float32)

        if self.resize_image!=None :
            dcm_img = self.resize_image(dcm_img)
            dcm_label=self.resize_image(dcm_label)
        
        dcm_img = self.normalize_image(dcm_img)

        if random.random() < 0.5 and self.trn: # it was > 0.5
            angle = random.randint(-90, 90)
            dcm_img = TF.rotate(dcm_img, angle)
            dcm_label = TF.rotate(dcm_label, angle)

            # second type: Apply deformable augmentations: GridDistortion
            augmented = self.deform_aug(image=dcm_img.numpy(), mask=dcm_label.numpy())
            dcm_img = torch.from_numpy(augmented['image']).float()
            dcm_label = torch.from_numpy(augmented['mask']).float()
            # Convert PyTorch tensors to NumPy arrays before augmentations
            dcm_img = np.ascontiguousarray(dcm_img.numpy())
            dcm_label = dcm_label.numpy()

            # # Third type: Apply horizontal and vertical flips:  Smooth aug: We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid.
            # augmented = self.transforms(image=dcm_img.squeeze(), mask=dcm_label.squeeze())
            # dcm_img = augmented['image'].float()
            # dcm_label = augmented['mask'].float().unsqueeze(0)


        meta_inf={'dcm_meta':dcm_meta.to_json_dict(),'img_path':dcm_img_path,'label_path':dcm_label_path}
     
        return dcm_img.float(), dcm_label.float(), meta_inf, boundary_map
        # return dcm_img.float(), dcm_label.float(), meta_inf

    def getLabelPath(self,dcm_img_path,dcm_meta):
        slice_number = int(dcm_meta[2, 3].value.split('.')[-1])

        dcm_label_path=dcm_img_path.split('/')
        dcm_label_path[-1]='segment.'+str(slice_number).zfill(3)+'.pgm'
        dcm_label_path.insert(-1,'auto')
        dcm_label_path='/'.join(dcm_label_path)
        # print(dcm_img_path,dcm_label_path)

        return dcm_label_path
    
    def split_train_valid_test(self,splits=[0.7,0.2,0.1]):
        assert len(splits) == 3, 'param splits must be list of three' 
        print(np.sum(splits))
        assert np.sum(splits) == 1.0, 'the total sum of the splits should be 1'
        train_patient_idx=int(len(self.patients_ids)*splits[0])
        valid_patient_idx=train_patient_idx+int(len(self.patients_ids)*splits[1])
        train_idx, valid_idx, test_idx=[],[],[]
        for i,path in enumerate(self.data_path):
            patient_id = path.split('/')[-5]
            patient_idx = self.patients_ids.index(patient_id)

            if(patient_idx<train_patient_idx):
                train_idx.append(i) 
            elif(patient_idx<valid_patient_idx):
                valid_idx.append(i)
            else:
                test_idx.append(i)
        x=len(train_idx)+len(valid_idx)+len(test_idx)
        # print(x)
        assert  x== len(self.data_path), "error in splitting the data check the implemenation of the split function"
        
        return train_idx, valid_idx, test_idx
    
    def split_train_valid_cross_validation(self, k):
        num_patients = len(self.patients_ids)
        assert num_patients >= 7, "There must be at least 7 patients for cross-validation."
        assert k >= 0 and k <= 6, "k must be between 0 and 6."
        
        valid_idx = []
        train_idx = []
        
        if k == 0:
            train_patients = self.patients_ids[:42]
            valid_patients = self.patients_ids[42:]
        elif k == 1:
            train_patients = self.patients_ids[:35] + self.patients_ids[42:]
            valid_patients = self.patients_ids[35:42]
        elif k == 2:
            train_patients = self.patients_ids[:28] +self.patients_ids[35:]
            valid_patients = self.patients_ids[28:35]
        elif k == 3:
            train_patients = self.patients_ids[:21] + self.patients_ids[28:]
            valid_patients = self.patients_ids[21:28]
        elif k == 4:
            train_patients = self.patients_ids[:14] + self.patients_ids[21:]
            valid_patients = self.patients_ids[14:21]
        elif k == 5:
            train_patients = self.patients_ids[:7] + self.patients_ids[14:]
            valid_patients = self.patients_ids[7:14]
        else:  # k == 6
            train_patients = self.patients_ids[7:]
            valid_patients = self.patients_ids[:7]
        
        valid_idx = [idx for idx, path in enumerate(self.data_path) if path.split('/')[-5] in valid_patients]
        train_idx = [idx for idx, path in enumerate(self.data_path) if path.split('/')[-5] in train_patients]
        
        # train_idx_list.append(train_idx)
        # valid_idx_list.append(valid_idx)
        x = len(train_idx) + len(valid_idx)
        assert x == len(self.data_path), "Error in splitting the data, check the implementation of the split function"
        
        return train_idx, valid_idx


    def set_train(self,train):
        self.trn=train
        
    def __len__(self):
        return len(self.data_path)

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
        
    def visualize_samples(self, num_images):
        print('get Samples:',num_images )
        samples = []
        rows = 2  # Number of rows in the grid
        cols = num_images  # Number of columns in the grid (assuming 2 images per row)

        # Create a new figure
        fig, axes = plt.subplots(rows, cols, figsize=(30, 5))
        
        for i in range(num_images):
            random_integer = random.randint(1, len(self.data_path))
            img,label= self.__getitem__(random_integer)
            axes[0,i].imshow(img.numpy().squeeze(),cmap='gray')
            axes[1,i].imshow(label.numpy().squeeze(),cmap='gray')
            samples.append((img,label))
        
        plt.tight_layout()
        plt.show()
        return samples
    
    def compute_boundary_map(self, label_mask):
        # Calculate the distance transform of the label_mask
        distance_map = distance_transform_edt(1 - label_mask)  
        ''' Use a distance transform algorithm to compute the distance from each pixel in the mask to the nearest 
            boundary. You can use the scipy.ndimage.distance_transform_edt function from the SciPy library for this 
            purpose'''
        # Invert the distance_map so that closer pixels have higher values
        boundary_map = 1 - (distance_map / (distance_map.max() + 1e-6))

        # You can further process or normalize the boundary_map as needed
        # apply a threshold to the map to ensure that only the pixels very close to the boundary are given high weights,

        return boundary_map
