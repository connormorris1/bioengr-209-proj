import pandas as pd
from pydicom import dcmread
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
import numpy as np
import random
import os
from matplotlib import pyplot as plt
from skimage.transform import rescale
from torchvision import models
from CustomImageDataset import CustomImageDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from train_and_test_loop import train_loop, test_loop
import time
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

all_dicoms = pd.read_csv("/home/cjmorris/dicom_practice/pt_ids/all_dicom_headers.csv")
all_dicoms = all_dicoms.dropna(subset='SeriesDescription')
svp_pids = list(pd.read_excel("/home/cjmorris/dicom_practice/pt_ids/95 SVP ground truth - HX.xlsx")['ip_patient_id'])
sa_df = all_dicoms[all_dicoms['SeriesDescription'].str.lower().str.contains('sa|short')]
sa_df = sa_df[~sa_df['SeriesDescription'].str.lower().str.contains('sag|montage|valsalva|scout|flow_quant|cemra|mpr|shortt1|delay|t2starmap|trufi_loc|loc')] #trufi_loc, loc scans have no time dimension, but are otherwise good I think
series_to_remove = ['tfl_loc_short-axis_iPAT','haste_sag','scout sa','scout SA','Cine SA montage','Delay_wideband SA_MAG','Delay_wideband SA_PSIR','MPR Ao sinuses of valsalva','MPR - sinuses of valsalva','T1Map_ShortT1_post1_19by60_MOCO','T1Map_ShortT1_post4_MOCO','T1Map_ShortT1_post3_27by60_MOCO','MPR LSA','MPR RSA','Thin mip rao gated sa_pf','Argus - EF SA Rpw','ShortIR_T1_T2map_bw1002_TR2.55_FA6_2','MPR Rt Basal','MPR gated SA','t2_tse_short axis','trufi2d_Real time SA_T-PAT3+']
# bh_cine_sa_192i_25ph_2beat_1011v4_Normalized - seem to be unable to view dicoms with slicer? 
sa_df = sa_df[~sa_df['SeriesDescription'].isin(series_to_remove)]
sa_df = sa_df[sa_df['num_frames'] >= 5]
sa_svp_df = sa_df[sa_df['PatientID'].isin(svp_pids)]
sa_svp_df['label'] = 1
sa_normal_df = sa_df[~sa_df['PatientID'].isin(svp_pids)]
sa_normal_df['label'] = 0
svp_pids = list(sa_svp_df['PatientID'].unique())
normal_pids = list(sa_normal_df['PatientID'].unique())
train_svp_pids = random.sample(svp_pids,int(np.round(len(svp_pids)*0.8)))
test_svp_pids = list(set(svp_pids) - set(train_svp_pids))
train_normal_pids = random.sample(normal_pids,int(np.round(len(normal_pids)*0.8)))
test_normal_pids = list(set(normal_pids) - set(train_normal_pids))



svp_test_csv = pd.read_csv('data_paths/svp_test.csv',header=None)
all_files = list(svp_test_csv[0])
all_dirs = list(set([os.path.dirname(file) for file in all_files]))
svp_test_df = sa_svp_df[sa_svp_df['sequence_path'].isin(all_dirs)]

normal_test_csv = pd.read_csv('data_paths/normal_test.csv',header=None)
all_files = list(normal_test_csv[0])
all_dirs = list(set([os.path.dirname(file) for file in all_files]))
normal_test_df = sa_normal_df[sa_normal_df['sequence_path'].isin(all_dirs)]



def block_upper_quartile(series_df,label):
    lower_loc_df = pd.DataFrame(columns = [0,1]) 
    all_loc = dict()
    for dir in series_df['sequence_path'].unique():
        all_dcm_filenames = os.listdir(dir)
        dcm = dcmread(os.path.join(dir,all_dcm_filenames[0]))
        loc = float(dcm.get((0x0021,0x1041),'Unknown').value)
        all_loc[loc] = dir
    all_loc_values = list(all_loc.keys())
    upper_quartile = np.percentile(all_loc_values,75)
    lower_loc_values = [x for x in all_loc_values if x < upper_quartile]
    for val in lower_loc_values:
        dir = all_loc[val]
        for file in os.listdir(dir):
            lower_loc_df.loc[len(lower_loc_df)] = [os.path.join(dir,file),label]
    return lower_loc_df

model_path = 'resnet_weights_longrun1.pth'
model = models.resnet18(pretrained = False)
model.fc = nn.Linear(model.fc.in_features, 1,bias=True) #***this model isn't built properly***
model.load_state_dict(torch.load(model_path))
criterion = nn.BCEWithLogitsLoss()
all_preds = []
all_labels = []
for pid in svp_test_df['PatientID'].unique():
    pdf = svp_test_df[svp_test_df['PatientID'] == pid]
    for series in pdf['SeriesDescription'].unique():
        sdf = pdf[pdf['SeriesDescription'] == series]
        for date in sdf['Date'].unique():
            ddf = sdf[sdf['Date'] == date]
            df_to_test = block_upper_quartile(ddf,1)
            dataloader = DataLoader(CustomImageDataset(df_to_test,224,False,file_is_df=True),batch_size=len(df_to_test)) 
            preds, labels = test_loop(test_dataloader,model,criterion,device='cpu')
            pt_pred = torch.max(preds)
            label = labels[0]
            print(pt_pred)
            print(label)
            all_preds.append(pt_pred)
            all_labels.append(label)

for pid in normal_test_df['PatientID'].unique():
    pdf = svp_test_df[normal_test_df['PatientID'] == pid]
    for series in pdf['SeriesDescription'].unique():
        sdf = pdf[pdf['SeriesDescription'] == series]
        for date in sdf['Date'].unique():
            ddf = sdf[sdf['Date'] == date]
            df_to_test = block_upper_quartile(ddf,0)
            dataloader = DataLoader(CustomImageDataset(df_to_test,224,False,file_is_df=True),batch_size=len(df_to_test)) 
            preds, labels = test_loop(test_dataloader,model,criterion,device='cpu')
            pt_pred = torch.max(preds)
            label = labels[0]
            print(pt_pred)
            print(label)
            all_preds.append(pt_pred)
            all_labels.append(label)

auc = roc_auc_score(all_labels,all_preds)
acc = accuracy_score(all_labels,all_preds)
prec = precision_score(all_labels,all_preds)
recall = recall_score(all_labels,all_preds)
print(auc, acc, prec, recall)