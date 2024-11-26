import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torchaudio import transforms as T

class SpecAugment(torch.nn.Module):
    def __init__(self, freq_mask=20, time_mask=50, freq_stripes=2, time_stripes=2, p=1.0):
        super().__init__()
        self.p = p
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.freq_stripes = freq_stripes
        self.time_stripes = time_stripes   
        self.specaugment = nn.Sequential(
            *[T.FrequencyMasking(freq_mask_param=self.freq_mask, iid_masks=True) for _ in range(self.freq_stripes)], 
            *[T.TimeMasking(time_mask_param=self.time_mask, iid_masks=True) for _ in range(self.time_stripes)],
            )
            
    def forward(self, audio):
        if self.p > torch.randn(1):
        # if self.p > 0:
            return self.specaugment(audio)
        else:
            return audio


class MelAugment(torch.nn.Module):
    def __init__(self, freq_mask=20, time_mask=50, freq_stripes=2, time_stripes=2, p=1.0):
        super().__init__()
        self.p = p
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.freq_stripes = freq_stripes
        self.time_stripes = time_stripes   
        self.specaugment = nn.Sequential(
            *[T.FrequencyMasking(freq_mask_param=self.freq_mask, iid_masks=True) for _ in range(self.freq_stripes)]
            )
            
    def forward(self, audio):
        # if self.p > torch.randn(1):
        if self.p > 0:
            return self.specaugment(audio)
        # else:
        #     return audio


        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32     # Downsampled ratio

        DEFAULT_OUT_DIM = 128 #for ssl embedding space dimension
        DEFAULT_NFFT = 1024
        DEFAULT_NMELS = 64
        DEFAULT_WIN_LENGTH = 1024
        DEFAULT_HOP_LENGTH = 323
        DEFAULT_FMIN = 50
        DEFAULT_FMAX = 8000

        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=323, 
        #     win_length=1024, window='hann', center=True, pad_mode='reflect', 
        #     freeze_parameters=True)
        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=16000, n_fft=1024, 
        #     n_mels=64, fmin=50, fmax=8000, ref=1.0, amin=1e-10, top_db=None, 
        #     freeze_parameters=True)
        # # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=40, time_stripes_num=2, 
        #     freq_drop_width=20, freq_stripes_num=2) # 2 2
        # self.bn0 = nn.BatchNorm2d(64)