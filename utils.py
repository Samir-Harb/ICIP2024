import pandas as pd
import glob
import pydicom as dicom
from pathlib import Path
import  numpy as np
import shutil
from tqdm import tqdm

def get_CTC_ds_as_csv(base_dir,seg_out_folders):
    seg_out_folders =['input_path']+seg_out_folders
    data_paths = glob.glob(base_dir+'/**/*.dcm',recursive=True)
    label_paths = []
    errors=[]
    slice_number_key = 0x020,0x013 # each dicom file has a value stored at the key 0x020,0x013 which represent its position in the sequence
    data_paths.sort()
    for temp_path in tqdm(data_paths):
        try:
            dcm = dicom.dcmread(temp_path)
            slice_number = int(dcm[slice_number_key].value)
            slice_number =str(slice_number).zfill(3)
            path = Path(temp_path)
            parent = path.parent.absolute()
            label_dic= dict.fromkeys(seg_out_folders,None)
            for seg_out in seg_out_folders[1:]:
                seg_path = parent/seg_out
                if (seg_path.exists()):
                    labelPath=next(seg_path.glob(f"*{slice_number}.pgm")).relative_to(base_dir)
                    label_dic[seg_out]=(labelPath)
            label_dic['input_path']=path.relative_to(base_dir)
            label_paths.append(label_dic)
        except Exception as e:
            print('Error occured during processing the data',e)
            errors.append(temp_path)
    print('saving dataset as csv')
    df = pd.DataFrame(label_paths)
    df.to_csv(base_dir+'/CTC_ds.csv')

def sortRenameDataset(base_dir,):
    
    data_paths = glob.glob(base_dir+'/**/*.dcm',recursive=True)
    label_paths = []
    errors=[]
    slice_number_key = 0x020,0x013 # each dicom file has a value stored at the key 0x020,0x013 which represent its position in the sequence
    data_paths.sort()
    for temp_path in tqdm(data_paths):
        try:
            dcm = dicom.dcmread(temp_path)
            slice_number = int(dcm[slice_number_key].value)
            slice_number =str(slice_number).zfill(3)
            newpath = temp_path.split('/')[0:-1]
            newpath.append(f'{slice_number}_.dcm')
            newpath = '/'.join(newpath)
            shutil.copyfile(temp_path, newpath)
            # temp_path.unlink()
        except Exception as e:
            print('Error occured during processing the data',e)
            errors.append(temp_path)
    # print('saving dataset as csv')
    # df = pd.DataFrame(label_paths)
    # df.to_csv(base_dir+'/CTC_ds.csv')
# sortRenameDataset('/media/hd2/Colon/Data_GT_Annotaion')

def cleanDs(base_dir,):
    
    data_paths = glob.glob(base_dir+'/**/*_.dcm',recursive=True)
    errors=[]
    data_paths.sort()
    for temp_path in tqdm(data_paths):
        try:
            newpath = temp_path.replace('_.dcm','.dcm')
            shutil.move(temp_path, newpath)
            # temp_path.unlink()
        except Exception as e:
            print('Error occured during processing the data',e)
            errors.append(temp_path)

cleanDs('/media/hd2/Colon/Data_GT_Annotaion')


def getPatiensIds(dataRoot,split =[0.7,0.2,0.1]):
    patientPaths = glob.glob(f'{dataRoot}/*/')
    paitents_ids = [path.split('/')[-2] for path in patientPaths ]
    paitents_ids.sort()
    trainSize = int(len(paitents_ids)* split[0])
    validSize = trainSize + int(len(paitents_ids)* split[1])
    testSize = len(paitents_ids) - validSize
    train_ids =paitents_ids[0:trainSize] 
    valid_ids =paitents_ids[trainSize:validSize]
    # valid_ids =paitents_ids[trainSize+3:trainSize+4]
    
    test_ids = paitents_ids[-testSize:]
    return train_ids, valid_ids, test_ids
