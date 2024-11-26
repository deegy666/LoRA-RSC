import os
import numpy as np
import torch
from args import args
from dataset_wav import ConfusionMatrix
labels=['Normal','Crackle','Wheeze','both']
import time
from cb import CB_loss


localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'
save_label = str_time

def train_epoch(model, train_loader, train_transform, criterion, optimizer, scheduler):
    
    if args.dataset == 'ICBHI':
        TP = [0, 0, 0 ,0]
        GT = [0, 0, 0, 0]
    epoch_loss = 0.0
    model.train()

    for data, target, _ in train_loader:
        data, target = data.to(args.device), target.to(args.device)

        with torch.no_grad():
           data_t = train_transform(data)
           
            # data_1=data.squeeze(dim = 1)
            # data_label=target.tolist()
            # data_data=data_1.tolist()
            # for i in range(len(data_label)):
            #     if data_label[i]==0:
            #         data_q=torch.tensor(data_data[i])
            #         data_q=data_q.to(args.device)
            #         data_q=data_q.unsqueeze(0)
            #         #data_q=val_transform(data_q)
            #         data_q=train_transform(data_q)
            #         data_data[i]=data_q.squeeze(0).tolist()
            #     else:
            #         data_w=torch.tensor(data_data[i])
            #         data_w=data_w.to(args.device)
            #         data_w=data_w.unsqueeze(0)
            #         data_w=mel_transform(data_w)
            #         data_data[i]=data_w.squeeze(0).tolist()
            # data_2=torch.tensor(data_data)
            # data_2=data_2.unsqueeze(-1)
            # data_2=data_2.permute(0, 3, 1, 2)
            # data_2=data_2.to(args.device)
            
        optimizer.zero_grad()

        output = model(data_t)
        #loss=criterion[0](output,target)+criterion[1](output,target)
        #loss=criterion(output,target)
        loss=CB_loss(target,output['ce_output'],[2063, 1215, 501, 363],4,"focal", 0.9, 1.0)+criterion[0](output['only_mel'],output['combine'],target)
        epoch_loss += loss.item()
        output_tensor=output['ce_output']
        _, labels_predicted = torch.max(output_tensor, dim=1)

        for idx in range(len(TP)):
            TP[idx] += torch.logical_and((labels_predicted==idx),(target==idx)).sum().item()
            GT[idx] += (target==idx).sum().item()
        
        
        loss.backward()
        optimizer.step()

    scheduler.step()

    epoch_loss = epoch_loss / len(train_loader)
    se = sum(TP[1:])/sum(GT[1:])
    sp = TP[0]/GT[0]
    icbhi_score = (se+sp)/2
    acc = sum(TP)/sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc

def val_epoch(model, val_loader, val_transform, criterion):

    if args.dataset == 'ICBHI':
        TP = [0, 0, 0 ,0]
        GT = [0, 0, 0, 0]
    elif args.dataset == 'SPRS':
        TP = [0, 0, 0 ,0, 0, 0, 0]
        GT = [0, 0, 0, 0, 0, 0, 0]

    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():

        for data, target, _ in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            
            
            output = model(val_transform(data))
            # loss = criterion(output, target)
            # epoch_loss += loss.item()
            
            _, labels_predicted = torch.max(output['ce_output'], dim=1)

            for idx in range(len(TP)):
                TP[idx] += torch.logical_and((labels_predicted==idx),(target==idx)).sum().item()
                GT[idx] += (target==idx).sum().item()

    # epoch_loss = epoch_loss / len(val_loader)
    se = sum(TP[1:])/sum(GT[1:])
    sp = TP[0]/GT[0]
    icbhi_score = (se+sp)/2
    acc = sum(TP)/sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc

def train_ce(model, train_loader, val_loader, train_transform, val_transform, criterion, optimizer, epochs, scheduler):

    train_losses = []; val_losses = []; train_se_scores = []; train_sp_scores = []; train_icbhi_scores = []; train_acc_scores = []; val_se_scores = []; val_sp_scores = []; val_icbhi_scores = []; val_acc_scores = []

    best_val_acc = 0
    best_icbhi_score = 0
    best_se = 0
    best_sp = 0
    best_epoch_acc = 0
    best_epoch_icbhi = 0
    start_time=time.time()
    for i in range(1, epochs+1):
        
        print(f"Epoch {i}")

        train_loss, train_se, train_sp, train_icbhi_score, train_acc = train_epoch(model, train_loader, train_transform, criterion, optimizer, scheduler)
        train_losses.append(train_loss); train_se_scores.append(train_se); train_sp_scores.append(train_sp); train_icbhi_scores.append(train_icbhi_score); train_acc_scores.append(train_acc)
        print(f"Train loss : {format(train_loss, '.4f')}\tTrain SE : {format(train_se, '.4f')}\tTrain SP : {format(train_sp, '.4f')}\tTrain Acc : {format(train_acc, '.4f')}\tTrain Score : {format(train_icbhi_score, '.4f')}")

        val_loss, val_se, val_sp, val_icbhi_score, val_acc = val_epoch(model, val_loader, val_transform, criterion)
        val_losses.append(val_loss); val_se_scores.append(val_se); val_sp_scores.append(val_sp); val_icbhi_scores.append(val_icbhi_score); val_acc_scores.append(val_acc)
        print(f"Val loss : {format(val_loss, '.4f')}\tVal SE : {format(val_se, '.4f')}\tVal SP : {format(val_sp, '.4f')}\tVal Acc : {format(val_acc, '.4f')}\tVal Score : {format(val_icbhi_score, '.4f')}")          
        #print(f"current best icbhi score is {format(best_icbhi_score, '.4f')} (se:{format(best_se, '.4f')} sp:{format(best_sp, '.4f')}) at epoch {best_epoch_icbhi}")
        if best_val_acc == 0:
            best_val_acc = val_acc

        if i == 1:
            best_icbhi_score = val_icbhi_score
            best_se = val_se
            best_sp = val_sp

        if best_icbhi_score < val_icbhi_score:
            best_epoch_icbhi = i
            best_icbhi_score = val_icbhi_score
            best_se = val_se
            best_sp = val_sp
            model_fname = os.path.join('./save', save_label)
            torch.save(model.state_dict(), model_fname)
        if best_val_acc < val_acc:
            best_epoch_acc = i
            best_val_acc = val_acc
    
        if val_icbhi_score > 0.62:
            model_fname = os.path.join('./save',f'{val_icbhi_score}')
            torch.save(model.state_dict(), model_fname)

    end_time=time.time()
    print('yunxingshijian',end_time-start_time)
    print(f"best icbhi score is {format(best_icbhi_score, '.4f')} (se:{format(best_se, '.4f')} sp:{format(best_sp, '.4f')}) at epoch {best_epoch_icbhi}")

    return train_losses, val_losses, train_se_scores, train_sp_scores, train_icbhi_scores, train_acc_scores, val_se_scores, val_sp_scores, val_icbhi_scores, val_acc_scores
