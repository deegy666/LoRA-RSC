import numpy as np
import torch
import torchaudio
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn as nn
from torchaudio import transforms as T
import torch.nn.functional as F
from utils1 import Normalize, Standardize
from augmentations import SpecAugment,MelAugment
from matplotlib import pyplot as plt
from torchaudio import transforms as T
from torch.utils.data import Dataset
import pandas as pd
import math
from transformers import AutoFeatureExtractor, AutoProcessor
import os
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.nn.functional as F
METADATA_CSV = 'metadata.csv'
DESIRED_DURATION = 8 # only 15 respiratory cycles have a length >= 8 secs, and the 5 cycles that have a length >= 9 secs contain artefacts towards the end
DESIRED_SR = 16000 # sampling rate
DEFAULT_OUT_DIM = 128 #for ssl embedding space dimension
DEFAULT_NFFT = 1024
DEFAULT_NMELS = 64
DEFAULT_WIN_LENGTH = 1024
DEFAULT_HOP_LENGTH = 323
DEFAULT_FMIN = 50
DEFAULT_FMAX = 8000
SPRS_CLASS_DICT = {'Normal' : 0, 'Fine Crackle' : 1, 'Wheeze' : 2, 'Coarse Crackle' : 3,'Wheeze+Crackle' : 4, 'Rhonchi' : 5, 'Stridor' : 6}
labels=['Normal','Crackle','Wheeze','both']
resize=Compose([
        Resize((1,128000))
    ])
melspec = T.MelSpectrogram(n_fft=DEFAULT_NFFT, n_mels=DEFAULT_NMELS, win_length=DEFAULT_WIN_LENGTH, hop_length=DEFAULT_HOP_LENGTH, f_min=DEFAULT_FMIN, f_max=DEFAULT_FMAX)
normalize = Normalize()
melspec = torch.nn.Sequential(melspec, normalize)
standardize = Standardize(device="cpu")
specaug = SpecAugment(freq_mask=20, time_mask=40, freq_stripes=2, time_stripes=2)
melaug = MelAugment(freq_mask=20, time_mask=40, freq_stripes=2, time_stripes=2)
train_transform = nn.Sequential(melspec, specaug, standardize)
val_transform = nn.Sequential(melspec, standardize)

class ConfusionMatrix(object):
    def __init__(self, num_classs : int,labels:list):
        self.matrix = np.zeros( (num_classs, num_classs) )
        self.num_classs=num_classs
        self.labels=labels
    def update(self,preds,labels):
        for p ,t in zip(preds,labels):
            self.matrix[p,t]+=1
    def plot(self):
        matrix=self.matrix
        print(matrix)
        plt.imshow(matrix,cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classs),self.labels,rotation=45)
        plt.yticks(range(self.num_classs), self.labels)
        plt.colorbar()
        plt.xlabel('True')
        plt.ylabel('pre')
        plt.title('Confusion matrix')
        thresh=matrix.max()/2
        for x in range(self.num_classs):
            for y in range(self.num_classs):
                info=int(matrix[y,x])
                plt.text(x,y,info,verticalalignment='center',
                         horizontalalignment='center' ,color='white' if info>thresh else 'black'        )

        plt.tight_layout()
        plt.show()

# ICBHI label mapping
"""
LABEL_N, LABEL_C, LABEL_W, LABEL_B = 0, 1, 2, 3
label 0 for normal respiration
label 1 for crackles
label 2 for wheezes
label 3 for both
"""

class ICBHI(Dataset):
    def __init__(self, data_path, split,
                  metadatafile=METADATA_CSV, duration=DESIRED_DURATION,
                    samplerate=DESIRED_SR, device="cpu",
                      fade_samples_ratio=16, pad_type="circular",
                        meta_label="", sigma=0.26, mu= 0.2,num_patches=0):
        self.processor = AutoFeatureExtractor.from_pretrained("./ast-finetuned-audioset-10-10-0.4593", max_length=500)
        self.data_path1=data_path
        self.data_path = os.path.join(self.data_path1, 'ICBHI_final_database')
        self.csv_path = os.path.join(self.data_path1, 'metadata.csv')
        self.split = split
        self.df = pd.read_csv(self.csv_path)
        if self.split == 'train':
            self.df = self.df[(self.df["split"] == self.split)]
        elif self.split == 'test':
            self.df = self.df[(self.df["split"] == self.split)]
        self.meta_label = meta_label
        self.duration = duration
        self.samplerate = samplerate
        self.targetsample = int(self.duration * self.samplerate)
        self.targetsample_wav=int(3 * self.samplerate)
        self.pad_type = pad_type
        self.device = device
        self.fade_samples_ratio = fade_samples_ratio
        self.fade_samples = int(self.samplerate/self.fade_samples_ratio)
        self.fade = T.Fade(fade_in_len=self.fade_samples, fade_out_len=self.fade_samples, fade_shape='linear')
        self.fade_out = T.Fade(fade_in_len=0, fade_out_len=self.fade_samples, fade_shape='linear')
        self.meta_label = meta_label
        self.num_patches = num_patches
        self.sigma = sigma
        self.mu = mu
        if self.meta_label != "":
            self.pth_path = os.path.join(self.data_path, "icbhi"+str(self.split)+'_duration'+str(self.duration)+"_metalabel-"+str(meta_label)+".pth")
        else:
            self.pth_path = os.path.join(self.data_path, "icbhi"+str(self.split)+'_duration'+str(self.duration)+".pth")

        if os.path.exists(self.pth_path):
            print(f"Loading dataset {self.split}...")
            pth_dataset = torch.load(self.pth_path)
            #self.data, self.labels, self.metadata_labels = pth_dataset['data'].to(self.device), pth_dataset['label'].to(self.device), pth_dataset['meta_label'].to(self.device)
            self.data,self.labels, self.metadata_labels = pth_dataset['data'], pth_dataset['label'], pth_dataset['meta_label']
            print(f"Dataset {self.split} loaded !")
        else:
            print(f"File {self.pth_path} does not exist. Creating dataset...")
            self.data, self.labels, self.metadata_labels = self.get_dataset()
            
            data_dict = {"data": self.data, "label": self.labels, "meta_label": self.metadata_labels}
            #self.data, self.labels, self.metadata_labels = self.data.to(self.device), self.labels.to(self.device), self.metadata_labels.to(self.device)
            print(f"Dataset {self.split} created !")
            torch.save(data_dict, self.pth_path)
            print(f"File {self.pth_path} Saved!")

    def get_random_patch(self, feature):

        def get_random_center(i):
            return np.random.randint(int(t * i / 2) + 1, int(t * (1 - i / 2))) / t

        _,t = feature.shape
        l = self.mu + self.sigma * np.random.randn(5 * self.num_patches)
        idx = [i >= 0.05 and i < 0.8 for i in l]
        l = l[idx][:self.num_patches]

        c = [get_random_center(i) for i in l]
        s, e = (c - l / 2) * t, (c + l / 2) * t
        s = [int(i) for i in s]
        features = []
        for i in range(0,self.num_patches,2):

            if s[i+1]<s[i]:
                start_time=s[i+1]
                end_time=s[i]
            else:
                start_time = s[i]
                end_time = s[i + 1]
            feature_s = feature[:,start_time:end_time]
            features.append(feature_s)
        feature=torch.cat((features[0],features[1],features[2]),dim=1)
        if feature.shape[-1] > self.targetsample:
            feature = feature[..., :self.targetsample]
        else:

            tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
            diff = self.targetsample - feature.shape[-1]
            tmp[..., diff // 2:feature.shape[-1] + diff // 2] = feature
            feature = tmp
        return feature


    def get_sample(self, i):

        ith_row = self.df.iloc[i]
        filepath = ith_row['filepath']
        filepath = os.path.join(self.data_path, filepath)
        onset = ith_row['onset']
        offset = ith_row['offset']
        bool_wheezes = ith_row['wheezes']
        bool_crackles = ith_row['crackles']
        #chest_loc = filepath[4:7]
        #rec_equip = ith_row['device']
        metalabel_colname = str(self.meta_label) + '_class_num'
        metadata_label = ith_row[metalabel_colname]
        #metadata_label = ith_row['sc_class_num']

        if not bool_wheezes:
            if not bool_crackles:
                label = 0
            else:
                label = 1
        else:
            if not bool_crackles:
                label = 2
            else:
                label = 3

        sr = librosa.get_samplerate(filepath)
        audio, _ = torchaudio.load(filepath, int(onset*sr), (int(offset*sr)-int(onset*sr)))
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.samplerate:
            resample = T.Resample(sr, self.samplerate)
            audio = resample(audio)

        return self.fade(audio), label, metadata_label


    def get_dataset(self):
        
        dataset = []
        
        
        labels = []
        metadata_labels = []
        #rec_equips = []
        
        for i in range(len(self.df)):
            audio, label, metadata_label = self.get_sample(i)
            audio_wav=audio
            if audio.shape[-1] > self.targetsample:
                audio = audio[...,:self.targetsample]
            else:
                if self.pad_type == 'circular':
                    ratio = math.ceil(self.targetsample / audio.shape[-1])
                    audio = audio.repeat(1, ratio)
                    audio = audio[...,:self.targetsample]
                    audio = self.fade_out(audio)
                elif self.pad_type == 'zero':
                    tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                    diff = self.targetsample - audio.shape[-1]
                    tmp[...,diff//2:audio.shape[-1]+diff//2] = audio
                    audio = tmp
            if audio_wav.shape[-1] > self.targetsample_wav:
                audio_wav = audio_wav[...,:self.targetsample_wav]
            else:
                if self.pad_type == 'circular':
                    ratio = math.ceil(self.targetsample_wav / audio_wav.shape[-1])
                    audio_wav = audio_wav.repeat(1, ratio)
                    audio_wav = audio_wav[...,:self.targetsample_wav]
                    audio_wav = self.fade_out(audio_wav)
            
            audio_numpy = audio.numpy().astype(np.float64)
            noise = np.random.randn(len(audio_numpy))
            augmented_data = audio_numpy + 0.3 * noise
                # Cast back to same data type
            audio_numpy = self.processor(augmented_data, sampling_rate= 16000, return_tensors='pt')['input_values']
            # dataset_wav.append(audio_wav)
            dataset.append(audio_numpy)

            # dataset.append(processor(audio, sampling_rate= 16000, return_tensors='pt')['input_values'])
            labels.append(label)
            metadata_labels.append(metadata_label)
            #rec_equips.append(rec_equip)
            
        return torch.unsqueeze(torch.vstack(dataset), 1), torch.tensor(labels), torch.tensor(metadata_labels)#rec_equips


    def __len__(self):
    
        return len(self.df)
    

    def __getitem__(self, idx):
        
        return self.data[idx], self.labels[idx], self.metadata_labels[idx]

