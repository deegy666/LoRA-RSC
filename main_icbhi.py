#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.optim import AdamW
from src.AST import AST
from src.AST_LoRA_ICBHI import AST_LoRA, AST_LoRA_ablation
from src.AST_adapters import AST_adapter, AST_adapter_hydra, AST_adapter_ablation
from src.AST_prompt_tuning import AST_Prefix_tuning, PromptAST, Prompt_config
from dataset.icbhi import ICBHI
from utils.engine_icbhi import eval_one_epoch, train_one_epoch
from torch.utils.data import DataLoader
import wandb
import argparse
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import time
import datetime
import yaml
import os
import copy


localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'
def get_args_parser():
    parser = argparse.ArgumentParser('RSC-LoRA',
                                     add_help=False)
    parser.add_argument('--data_path', type= str, help = 'Path to the location of the dataset.',default='./data/ICBHI')
    parser.add_argument('--seed', default= 10)   #Set it to None if you don't want to set it.
    parser.add_argument('--device', type= str, default= 'cuda:0', 
                        help='device to use for training/testing')
    parser.add_argument('--num_workers', type= int, default= 6)
    parser.add_argument('--model_ckpt_AST', default= './ast-finetuned-audioset-10-10-0.4593')
    parser.add_argument('--save_best_ckpt', type= bool, default= True)
    parser.add_argument('--output_path', type= str, default= '/checkpoints')
    
    parser.add_argument('--dataset_name', type= str, choices = ['ICBHI'], default='ICBHI')
    parser.add_argument('--method', type= str, choices = ['linear', 'full-FT', 'adapter', 'prompt-tuning', 'prefix-tuning', 
                                                          'LoRA'], default='LoRA')
    # Adapter params.
    parser.add_argument('--seq_or_par', default = 'sequential', choices=['sequential','parallel'])
    parser.add_argument('--reduction_rate_adapter', type= int, default= 64)
    parser.add_argument('--adapter_type', type= str, default = 'Pfeiffer', choices = ['Houlsby', 'Pfeiffer'])
    parser.add_argument('--apply_residual', type= bool, default=True)
    parser.add_argument('--adapter_block', type= str, default='convpass', choices = ['bottleneck', 'convpass'])
   
    
    # Params for adapter ablation studies.
    parser.add_argument('--is_adapter_ablation', default= False)
    parser.add_argument('--befafter', type = str, default='after', choices = ['after','before'])
    parser.add_argument('--location', type = str, default='FFN', choices = ['MHSA','FFN'])
    
    
    # LoRA params.
    parser.add_argument('--reduction_rate_lora', type= int, default=96) #64  768 384 192 96 48 24 12
    parser.add_argument('--alpha_lora', type= int, default= 8)
    parser.add_argument('--lora_type', type = str, default = 'kv')

    # Params for LoRA ablation studies.
    parser.add_argument('--is_lora_ablation', type= bool, default= False)
    parser.add_argument('--lora_config', type = str, default = 'Wq,Wv', choices = ['Wq','Wq,Wk','Wq,Wv','Wq,Wk,Wv,Wo'])
    
    # Prefix-tuning params.
    parser.add_argument('--prompt_len_pt', type= int, default =96)
    
    # Prompt-tuning params.
    parser.add_argument('--prompt_len_prompt', type= int, default = 200)
    parser.add_argument('--is_deep_prompt', type= bool, default= True)
    parser.add_argument('--drop_prompt', default= 0.)
    
    # Few-shot experiments.
    parser.add_argument('--is_few_shot_exp', default = False)
    parser.add_argument('--few_shot_samples', default = 64)
    
    # WANDB args. 
    parser.add_argument('--use_wandb', type= bool, default= False)
    parser.add_argument('--project_name', type= str, default= '')
    parser.add_argument('--exp_name', type= str, default= '')
    parser.add_argument('--entity', type= str, default= '')
    
    return parser

def main(args):
    
    start_time = time.time()
    
    if args.use_wandb:
        wandb.init(project= args.project_name, name= args.exp_name,  entity= args.entity,
                   )
    print(args) 
    
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility (if desired).
    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed) 
    
    with open('./train.yaml', 'r') as file:
        train_params = yaml.safe_load(file)
    
    
    if args.dataset_name == 'ICBHI':
        max_len_AST = train_params['max_len_AST_ICBHI']
        num_classes = train_params['num_classes_ICBHI']
        batch_size = train_params['batch_size_ICBHI']
        epochs = train_params['epochs_ICBHI']
    else:
        raise ValueError('The dataset you chose is not supported as of now.')
        
    
    if args.method == 'prompt-tuning':
        final_output = train_params['final_output_prompt_tuning']
    else:
        final_output = train_params['final_output']
    
    
   
        
    # DATASETS
    train_data = ICBHI(data_path=args.data_path, metadatafile='metadata.csv', duration=8, split='train', device=args.device, samplerate=16000, pad_type='circular', meta_label='sa')
    test_data = ICBHI(data_path=args.data_path, metadatafile='metadata.csv', duration=8, split='test', device=args.device, samplerate=16000, pad_type='circular', meta_label='sa')
        
    
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
    test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= False, num_workers= args.num_workers, pin_memory= True, drop_last= False,)

        # MODEL definition.
        
    method = args.method
        
    if args.is_adapter_ablation:
            model = AST_adapter_ablation(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, seq_or_par= args.seq_or_par, location= args.location, adapter_block= args.adapter_block, before_after= args.befafter, model_ckpt= args.model_ckpt_AST).to(device)
            lr= train_params['lr_adapter']
    elif args.is_lora_ablation:
            model = AST_LoRA_ablation(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, rank= args.reduction_rate_lora, alpha= args.alpha_lora, lora_config= args.lora_config, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_LoRA']
    elif method == 'full-FT':
            model = AST(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_fullFT']
    elif method == 'linear':
            model = AST(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, model_ckpt= args.model_ckpt_AST).to(device)
            # Freeze the AST encoder, only the classifier is trainable.
            model.model.requires_grad_(False)
            # LN is trainable.
            model.model.layernorm.requires_grad_(True)
            lr = train_params['lr_linear']
        
    elif method == 'LoRA':
            model = AST_LoRA(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, rank= args.reduction_rate_lora, alpha= args.alpha_lora, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_LoRA']
    elif method == 'prefix-tuning':
            model = AST_Prefix_tuning(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, num_tokens= args.prompt_len_pt, patch_size= train_params['patch_size'], hidden_size= train_params['hidden_size'], model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_prompt']
    elif method == 'prompt-tuning':
            prompt_config = Prompt_config(NUM_TOKENS= args.prompt_len_prompt, DEEP= args.is_deep_prompt, DROPOUT= args.drop_prompt, FINAL_OUTPUT=final_output)
            model = PromptAST(prompt_config= prompt_config, max_length= max_len_AST, num_classes= num_classes, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_prompt']
    elif method == 'adapter':
            model = AST_adapter(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, adapter_type= args.adapter_type, seq_or_par= args.seq_or_par, apply_residual= args.apply_residual, adapter_block= args.adapter_block, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_adapter']
    else:
            raise ValueError('The method you chose is not supported as of now.')
            
        
        # PRINT MODEL PARAMETERS
    n_parameters = sum(p.numel() for p in model.parameters())
    print('Number of params of the model:', n_parameters)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    print('Number of trainable params of the model:', n_parameters)
        
    print(model)
        
        
    if method == 'linear': # LR of the backbone to finetune must be quite smaller than the classifier.
            optimizer = AdamW([{'params': model.model.parameters()}, {'params': model.classification_head.parameters(),'lr': 1e-3}],lr= lr,
                                  betas= (0.9,0.98), eps= 1e-6, weight_decay= train_params['weight_decay'] )
    else:
            optimizer = AdamW(model.parameters(), lr= lr ,betas= (0.9, 0.98),eps= 1e-6, weight_decay= train_params['weight_decay'])

        # weights = torch.tensor([2063, 1215, 501, 363], dtype=torch.float32) 
        # weights = weights / weights.sum()
        # weights = 1.0 / weights
        # weights = weights / weights.sum()
        # weights = weights.to(args.device)     #Move weights to GPU
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*(epochs))

    print(f"Start training for {epochs} epochs")
        
    best_score = 0.
    best_se = 0.
    best_sp = 0.
        
    for epoch in range(epochs):
            train_loss, train_acc, train_se, train_sp, train_icbhi_score, tacc= train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion)
            print(f"Trainloss at epoch {epoch}: {train_loss}")
           
            
            val_loss, val_acc, val_se, val_sp, val_icbhi_score, vacc = eval_one_epoch(model, test_loader, device, criterion)
            
            if val_icbhi_score > best_score:
                best_score = val_icbhi_score
                best_se = val_se
                best_sp = val_sp
                best_params = model.state_dict()
                
                #if args.save_best_ckpt:
                    #lora_r = 768/(args.reduction_rate_lora)
                    #torch.save(best_params, os.getcwd() + args.output_path + f'/bestmodel{args.method,str_time} + f'{best_score}')
                    #torch.save(best_params, os.getcwd() + args.output_path + f'/{best_score,args.method,str_time,args.lora_type,lora_r}')    
                
            print(f"Train SE : {format(train_se, '.4f')}\tTrain SP : {format(train_sp, '.4f')}\tTrain Acc : {format(tacc, '.4f')}\tTrain Score : {format(train_icbhi_score, '.4f')}")
            print(f"Val SE : {format(val_se, '.4f')}\tVal SP : {format(val_sp, '.4f')}\tVal Acc : {format(vacc, '.4f')}\tVal Score : {format(val_icbhi_score, '.4f')}")                    
           
            current_lr = optimizer.param_groups[0]['lr']
            print('Learning rate after initialization: ', current_lr)
            
            if args.use_wandb:
                wandb.log({"train_loss": train_loss, "valid_loss": val_loss,
                           "train_accuracy": train_acc, "val_accuracy": val_acc,
                           "lr": current_lr, }
                          )
        
    best_model = copy.copy(model)
    best_model.load_state_dict(best_params)
        
    
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    #lora_r = 768/(args.reduction_rate_lora)
    #torch.save(best_params, os.getcwd() + args.output_path + f'/bestmodel{args.method,str_time} + f'{best_score}')
    torch.save(best_params, os.getcwd() + args.output_path + f'q{str_time}')
    print('Training time {}'.format(total_time_str))
    print(f"best icbhi score is {format(best_score, '.4f')} (se:{format(best_se, '.4f')} sp:{format(best_sp, '.4f')})")
    if args.use_wandb:
        wandb.finish()

if __name__=="__main__":
    parser = argparse.ArgumentParser('RSC-LoRA',
                                    parents=[get_args_parser()])
    args = parser.parse_args(args=[])
    main(args)