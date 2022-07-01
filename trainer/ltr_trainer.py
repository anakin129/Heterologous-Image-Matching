import os
import torch
import time
from collections import OrderedDict
from .base_trainer import BaseTrainer
from utils.config import cfg
from utils.tensorboad import TensorboardWriter
from utils.stats import AverageMeter,StatValue
from utils.tensor_dict import TensorDict
import logging

class LTRTrainer(BaseTrainer):
    def __init__(self,actor,loaders,optimizer,lr_scheduler=None):
      
        """
        args:
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
        """
        super().__init__(actor,loaders,optimizer,lr_scheduler)
          # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

    def train_epoch(self):
        # do one epoch for each loader
        for loader in self.loaders:
            self.cycle_dataset(loader)
        self._stats_new_epoch()

    def cycle_dataset(self,loader):
        # do a cycle of training or validation
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()

        for i,data in enumerate(loader,1):
            if not data.__class__.__name__=='TensorDict':
                data=TensorDict(data)
            data=data.to(self.device)
            data['epoch']=self.epoch
            
            #forward pass
            loss,stats=self.actor(data)
            #backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            #update statisics
            batch_size=data['search_images'].shape[0]

            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
    
    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size): 
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % cfg.TRAIN.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            #print(print_str[:-5])
            logger=logging.getLogger('%s'%cfg.TRAIN.log)
            logger.info(print_str[:-5])
    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_last_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()