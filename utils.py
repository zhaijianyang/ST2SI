import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.optim import AdamW
import torch.nn as nn
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import logging
from accelerate.logging import MultiProcessAdapter

MAX_VAL = 1e4
    
def now_time():
    return '[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']: '

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores, labels):
        labels = labels.squeeze()
        
        try:
            loss = self.ce(scores, labels).item()
        except:
            print(scores.size())
            print(labels.size())
            loss = 0.0
        
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC

        return res + [loss]

def create_optimizer_and_scheduler(model: nn.Module, num_train_optimization_steps, args):
    
    base_param_optimizer = []
    linear_param_optimizer = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'decoder' in name:
                base_param_optimizer.append((name, param))
            else:
                linear_param_optimizer.append(param)
    
    # param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in base_param_optimizer if not any(nd in n for nd in no_decay)], 'lr': args.base_lr, 'weight_decay': args.base_weight_decay},
        {'params': [p for n, p in base_param_optimizer if any(nd in n for nd in no_decay)], 'lr': args.base_lr, 'weight_decay': 0.0},
        {'params': linear_param_optimizer, 'lr': args.linear_lr, 'weight_decay': args.linear_weight_decay}
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)

    return optimizer, scheduler

def get_logger(log_file, name=None, filemode='w'):

    logging.basicConfig(filename = log_file, filemode = filemode, level = logging.INFO)
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())

    return MultiProcessAdapter(logger, {})