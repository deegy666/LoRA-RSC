# LoRA-RSC
This is the code repository for paper "Low-rank Adaptation Method for Respiratory Sound Classification: A necessary road towards Large Models".

## Prepare dataset and pre-training model
#### dataset
 You can download it from the official ICBHI website https://bhichallenge.med.auth.gr/
#### AST pretrain
You can download it from the https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
## how to run
#### requirements
numpy==1.22.3 
torch==1.13.1
torchaudio==0.13.1
torchvision==0.14.1
transformers==4.28.1
librosa==0.9.2
### 
python main_icbhi.py  
You can modify the training parameters as needed. If you want checkpoints, you can contact me by email.

### Acknowledgments 
Thanks to Umberto Cappellazzo and others for the code.
