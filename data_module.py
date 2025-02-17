import os
import gzip
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

class Pose2MuscleDataset(Dataset):
    def __init__(self, path):
        self.data = []
        # load data
        print("data load in memory")
        with gzip.open(path, "rb") as rb_f:
            self.data = pickle.load(rb_f)
    
    def __len__(self):

        return

    def __getitem__(self, idx):

        return

class Pose2MuscleDataset(Dataset):
    def __init__(self, datasplit_path, target_muscle): # 불러올 데이터 세팅
        muscle_id = {'ES': 0, 'UT': 1, 'BB': 2, 'FDS': 3, 'ED': 4, 'BF': 5, 'VL': 6}
        
        self.subject_info_keys = ['age', 'height', 'weight', 'muscle', 'fat', 'arm_len', 'leg_len', 'load']
        # Train / Test에서 사용하는 subject에 따라 변함
        self.subject_info_values = [[24, 174, 70.1, 27.8, 20.3, 25.5, 39.5],
                                    [23, 177, 67.0, 30.0, 13.4, 28.0, 40.0],
                                    [24, 169, 69.6, 32.0, 13.5, 28.0, 40.5],
                                    [23, 175, 94.0, 35.5, 31.5, 26.5, 39.5],
                                    [20, 183, 86.9, 36.7, 21.8, 29.0, 42.5],
                                    [25, 173, 70.0, 35.0, 15.4, 26.0, 40.0],
                                    [25, 173, 77.4, 37.4, 12.6, 26.0, 42.5], 
                                    [25, 169, 68.0, 29.1, 15.7, 25.0, 40.0], 
                                    [24, 174, 83.2, 37.2, 18.0, 25.0, 40.0], 
                                    [25, 170, 62.3, 25.3, 16.9, 26.0, 36.0],
                                    [25, 180, 106.8, 49.3, 21.8, 25.0, 40.0],
                                    [23, 170, 56.2, 29.5, 4.3, 25.0, 40.0],
                                    [24, 173, 68.1, 32.7, 10.5, 26.0, 37.0],
                                    [24, 176, 79.2, 35.6, 16.7, 26.0, 40.0],
                                    [23, 176, 67.8, 30.4, 13.9, 26.0, 40.0],
                                    [23, 166, 52.7, 25.4, 7.1, 24.0, 38.0],]
        
        self.scaler = StandardScaler()
        self.subject_info_values = self.scaler.fit_transform(self.subject_info_values)

        # 포즈랑 근전도 로드
        self.datasplit_path = datasplit_path # './datasplits'

        self.data = []
        with open(self.datasplit_path, 'r') as r:
            lines = r.readlines()

        for _ in lines:
            x = {}
            filepath = _.strip('\n')
            x['filename'] = filepath
            x['subject'] = filepath.split('/')[2]

            x['subject_info'] = self.subject_info_values[int(x['subject'][7:])]
            weight = x['subject_info'][2]
            load = int(filepath.split('/')[3].split('kg')[0]) / weight
            x['subject_info'] = np.append(x['subject_info'].reshape(1, -1), np.array([load]))

            x['pose3d'] = np.load(f'{filepath}/pose3d.npy')
            
            emg = torch.Tensor(np.load(f'{filepath}/emg_values.npy', allow_pickle=True).astype(np.float32))
            emg_list = []
            for muscle in target_muscle:
                emg_list.append(emg[:, muscle_id[muscle]].unsqueeze(1))
            x['emg'] = torch.cat(emg_list, dim=1)
            
            self.data.append(x)
        
        print(f"Number of data loaded in memory: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # 초기화한 데이터에서 리턴할 데이터 필드 세팅
        filename = self.data[idx]['filename']
        
        subject = torch.Tensor(self.data[idx]['subject_info'])
        
        pose3d = torch.Tensor(self.data[idx]['pose3d'])
        
        emg_values = self.data[idx]['emg']
        
        return filename, subject, pose3d, emg_values

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_path = args['train_path']
        self.valid_path = args['valid_path']
        self.test_path = args['test_path']
    
    
    def setup(self, stage=None):
        self.train_dataset = Pose2MuscleDataset(self.train_path, self.args['target_muscle'])
        self.valid_dataset = Pose2MuscleDataset(self.valid_path, self.args['target_muscle'])
        self.test_dataset = Pose2MuscleDataset(self.test_path, self.args['target_muscle'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers'])
    
    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers'])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers'])