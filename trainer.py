import torch
from tqdm import tqdm
import os
import torch.distributed as dist
import numpy as np

from utils import AverageMeter, AverageMeterSet, Ranker, now_time, create_optimizer_and_scheduler

class Trainer:
    def __init__(self, args, accelerator, model, train_loader, dev_loader, test_loader):
        self.args = args
        self.logger = args.logger
        self.accelerator = accelerator
        
        num_train_optimization_steps = (len(train_loader) * args.num_train_epochs) // args.gradient_accumulation_steps
        optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
        
        self.model, self.train_loader, self.dev_loader, self.test_loader, self.optimizer, self.scheduler = accelerator.prepare(\
            model, train_loader, dev_loader, test_loader, optimizer, scheduler)
        
    def save_checkpoints(self, save_path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), save_path)
        
    def load_checkpoints(self, load_path):
        self.accelerator.wait_for_everyone()

        self.logger.info(now_time() + f'Load best model: {load_path}')
        if dist.is_initialized():
            self.model.module.load_state_dict(torch.load(load_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(load_path, map_location='cpu'))
    
    def predict(self, data_loader=None):
        
        if data_loader is None:
            data_loader = self.dev_loader

        ranker = Ranker(self.args.metric_ks)
        average_meter_set = AverageMeterSet()
        
        self.model.eval()
        with torch.no_grad():
            
            for batch in tqdm(data_loader, ncols=100, desc='Evaluate', disable=(not self.accelerator.is_local_main_process)):

                input_ids, attention_mask, target_seq_ids, labels = batch
                
                scores = self.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    )
                
                scores, labels = self.accelerator.gather_for_metrics((scores, labels))

                res = ranker(scores, labels)

                metrics = {}
                for i, k in enumerate(self.args.metric_ks):
                    metrics["NDCG@%d" % k] = res[2*i]
                    metrics["Recall@%d" % k] = res[2*i+1]
                metrics["MRR"] = res[-3]
                metrics["AUC"] = res[-2]

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                    
        average_metrics = average_meter_set.averages()

        return average_metrics
        
    def train_one_epoch(self, epoch):

        self.model.train()
        
        loss_meter = AverageMeter()
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=100, disable=(not self.accelerator.is_local_main_process))
        
        for step, batch in pbar:
            with self.accelerator.accumulate(self.model):
                
                input_ids, attention_mask, target_seq_ids, labels = batch
                
                loss = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=target_seq_ids,
                                )

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                loss_meter.update(loss.item())
                pbar.set_description(f"Epoch: {epoch + 1}, train loss: {loss_meter.avg:.5f}, lr: {self.scheduler.get_last_lr()[0]:.7f}")
                
        self.logger.info(now_time() + f'Epoch: {epoch + 1}, training loss: {loss_meter.avg:.5f}')
        
    def train(self):
        
        best_target = float('-inf')
        patient = self.args.patient

        for epoch in range(self.args.num_train_epochs):
            
            self.train_one_epoch(epoch)
            
            if (epoch + 1) % self.args.save_interval == 0:
                self.logger.info(now_time() + f'Save model epoch {epoch + 1}')
                save_path = os.path.join(self.args.output_path, f'epoch_{epoch + 1}.bin')
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_local_main_process:
                    self.save_checkpoints(save_path)
            
            if (epoch + 1) % self.args.interval == 0 and (epoch + 1) > self.args.skip_valid:

                dev_metrics = self.predict()
                
                self.logger.info(now_time() + f'Epoch: {epoch + 1}. Dev set: {dev_metrics}')

                if dev_metrics[self.args.valid_metric] > best_target:
                    best_target = dev_metrics[self.args.valid_metric]
                    patient = self.args.patient
                    self.logger.info(now_time() + 'Save the best model.')
                    save_path = os.path.join(self.args.output_path, 'pytorch_model.bin')
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_local_main_process:
                        self.save_checkpoints(save_path)
                else:
                    patient -= 1
                    if patient == 0:
                        break

        self.load_checkpoints(load_path=save_path)
        test_metrics = self.predict(self.test_loader)
        self.logger.info(now_time() + f'==Test set==: {test_metrics}\n')