#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train(True)
    
    TP = [0, 0, 0 ,0]
    GT = [0, 0, 0, 0]
    loss = 0.
    correct = 0
    total = 0
    
    for idx_batch, (data, target, _  ) in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        outputs = model(data)
        _, labels_predicted = torch.max(outputs, dim=1)
        loss_batch = criterion(outputs,target)     
        loss += loss_batch.detach().item()
        total += len(data)
        correct += (target==outputs.argmax(dim=-1)).sum().item()

        for idx in range(len(TP)):
            TP[idx] += torch.logical_and((labels_predicted==idx),(target==idx)).sum().item()
            GT[idx] += (target==idx).sum().item()
        
        
        loss_batch.backward()
        optimizer.step()
        scheduler.step()
    
    loss /= len(loader)
    accuracy = correct/total
    se = sum(TP[1:])/sum(GT[1:])
    sp = TP[0]/GT[0]
    icbhi_score = (se+sp)/2
    acc = sum(TP)/sum(GT)
    
    return loss, accuracy, se, sp, icbhi_score, acc

def eval_one_epoch(model, loader, device, criterion):
    
    loss = 0.
    correct = 0
    total = 0  
    
    model.eval()
    TP = [0, 0, 0 ,0]
    GT = [0, 0, 0, 0]
    with torch.inference_mode():
        for idx_batch, (data, target, _  ) in enumerate(loader): 

            
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            loss_batch = criterion(outputs, target)
            _, labels_predicted = torch.max(outputs, dim=1)
            loss += loss_batch.detach().item()
            total += len(data)
            correct += (target==outputs.argmax(dim=-1)).sum().item()
            for idx in range(len(TP)):
                TP[idx] += torch.logical_and((labels_predicted==idx),(target==idx)).sum().item()
                GT[idx] += (target==idx).sum().item()
        
        loss /= len(loader)
        accuracy = correct/total
        se = sum(TP[1:])/sum(GT[1:])
        sp = TP[0]/GT[0]
        icbhi_score = (se+sp)/2
        acc = sum(TP)/sum(GT)
        
        
    return loss, accuracy, se, sp, icbhi_score, acc