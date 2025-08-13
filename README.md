# LoRA-RSC
This is the code repository for paper "Low-rank Adaptation Method for Respiratory Sound Classification: A necessary road towards Large Models", accepted by ICASSP 2025. 📑 <a href="https://ieeexplore.ieee.org/abstract/document/10889965/">Paper</a>

## Overview
Automatic deep learning-based classification of respiratory sounds is important for the diagnosis of lung diseases. In recent years, many researchers have used pre-trained models to learn more comprehensive features of respiratory sounds. However, as the models become larger, the challenges of long training time and high memory consumption are often faced when fine-tuning the pre-trained model.To alleviate this problem, we apply **Low Rank Adaptation (LoRA) to freeze most of the pre-trained model weights and inject the trainable rank decomposition matrices into each layer of the Transformer Encoder**. Experiments on the ICBHI 2017 dataset show that the LoRA method reduces the number of trainable parameters by 85M and improves the ICBHI score by `0.94%` compared to full fine-tuning. This suggests that the LoRA method has great promise for pre-training respiratory sound classification models.

## 📊 Comparison of Our Method with State-of-the-Art on the ICBHI Dataset

**Comparison of Parameter-Efficient Fine-Tuning Methods on the ICBHI Dataset**

| Method                       | \#Train Params (M) | $S_e$(%) | $S_p$(%) | Score(%) |
|------------------------------|--------------------|----------|----------|----------|
| Full fine-tuning (Baseline)  | 85.74              | 36.70    | 82.84    | 59.77    |
| Adapter                      | 0.454              | 43.50    | 75.62    | 59.56    |
| Prefix                       | 0.890              | 39.68    | 74.41    | 57.05    |
| Prompt                       | 1.830              | 49.02    | 68.27    | 58.65    |
| **LoRA (Ours)**              | **0.299**          | **36.11**| **85.31**| **60.71**|


## 🚀 Prepare dataset and pre-training model
#### 1. Download dataset
 ICBHI 2017 Respiratory Sound Database [Official Challenge Page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge), set Path: After downloading and extracting the dataset, set the path to the data folder using the --data_folder argument. For example:```args.data_path = './data/ICBHI```
### 2. Download AST pre-training model
You can download the pretrained AST model from [Hugging Face](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593).
After downloading, you can set the path to the checkpoint file by assigning it to args.audioset_ckpt. For example:```args.audioset_ckpt = '/path/to/ast-finetuned-audioset-10-10-0.4593.pth'```

## 🧠 How to run
#### requirements
``` numpy==1.22.3 ,torch==1.13.1, torchaudio==0.13.1, torchvision==0.14.1, transformers==4.28.1, librosa==0.9.2``` 
### 
```python main_icbhi.py``` 
You can modify the training parameters as needed. If you want checkpoints, you can contact me by email.


## 📚 Citation
Please cite related papers if you use LoRA-RSC.

```
@inproceedings{dong2025low,
  title={Low-rank Adaptation Method for Respiratory Sound Classification: A necessary road towards Large Models},
  author={Dong, Gaoyang and Shen, Yufei and Wang, Jianhong and Xie, Shunwang and Zhang, Minghui and Sun, Ping},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## 🙏 Acknowledgments 

Thanks to Umberto Cappellazzo and others for the code.
