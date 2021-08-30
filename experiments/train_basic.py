#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Basic training script
'''

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch
import torch.nn.parallel
import torch.optim as optim
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from dataloaders import point_cloud_dataset_train, gen_rotation_matrix
from models.autoencoder import PointCloudAutoencoder
from util.option_handler import TrainOptionHandler
from tensorboardX import SummaryWriter
import time

import warnings
warnings.filterwarnings("ignore")

def main():
    # Build up an autoencoder
    t = time.time()
    print(torch.cuda.device_count(), "GPUs will be used for training.")
    device = torch.device("cuda:0")
    ae = PointCloudAutoencoder(opt)
    ae = torch.nn.DataParallel(ae)
    ae.to(device)

    # Create folder to save trained models
    if not os.path.exists(opt.exp_name):
        os.makedirs(opt.exp_name)

    # Take care of the dataset
    _, train_dataloader, _, _ = point_cloud_dataset_train(opt.dataset_name, opt.num_points, opt.batch_size, opt.train_split)

    # Create a tensorboard writer
    if opt.tf_summary: writer = SummaryWriter(comment=str.replace(opt.exp_name,'/','_'))

    # Setup the optimizer and the scheduler
    lr = opt.lr
    optimizer = optim.Adam(ae.parameters(), lr=lr[0], betas=(opt.optim_args[0], opt.optim_args[1]), weight_decay=opt.optim_args[2])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr[1], gamma=lr[2], last_epoch=-1)

    # Load a check point if given
    if opt.checkpoint != '':
        checkpoint = torch.load(opt.checkpoint)
        ae.module.load_state_dict(checkpoint['model_state_dict'])
        if opt.load_weight_only == False:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Existing model %s loaded.\n" % (opt.checkpoint))
    epoch_start = scheduler.state_dict()['last_epoch']

    # Start training
    n_iter = 0
    for epoch in range(epoch_start, opt.n_epoch):
        ae.train()
        print('Training at epoch %d with lr %f' % (epoch, scheduler.get_lr()[0]))

        train_loss_sum = 0
        batch_id = 0
        not_end_yet = True
        it = iter(train_dataloader)
        len_train = len(train_dataloader)

        # Iterates the training process
        while not_end_yet == True:
            points, _ = next(it)
            not_end_yet = batch_id + 1 < len_train
            if(points.size(0) < opt.batch_size):
                break

            # Data augmentation
            points = points.cuda()
            if opt.augmentation == True:
                if opt.augmentation_theta != 0 or opt.augmentation_rotation_axis != 0: # rotation augmentation
                    rotation_matrix = gen_rotation_matrix(theta=opt.augmentation_theta, 
                        rotation_axis=opt.augmentation_rotation_axis).unsqueeze(0).cuda().expand(opt.batch_size, -1 , -1)
                    points = torch.bmm(points, rotation_matrix)
                if opt.augmentation_max_scale - opt.augmentation_min_scale > 1e-3: # scaling augmentation
                    noise_scale = np.random.uniform(opt.augmentation_min_scale, opt.augmentation_max_scale)
                    points *= noise_scale
                if (opt.augmentation_flip_axis >= 0) and (np.random.rand() < 0.5): # fliping augmentation
                    points[:,:,opt.augmentation_flip_axis] *= -1
            optimizer.zero_grad()

            # Forward and backward computation
            rec = ae(points)
            rec = rec["rec"]
            train_loss = ae.module.xyz_loss(points, rec)
            train_loss.backward()
            optimizer.step()

            # Log the result
            train_loss_sum += train_loss.item()
            if batch_id % opt.print_freq == 0:
                elapse = time.time() - t
                print('    batch_no: %d/%d, time/iter: %f/%d, train_loss: %f' % 
                    (batch_id, len_train, elapse, n_iter, train_loss.item()))
                if opt.tf_summary: writer.add_scalar('train/batch_train_loss', train_loss.item(), n_iter)

            if opt.pc_write_freq > 0 and batch_id % opt.pc_write_freq == 0:
                labels = np.concatenate((np.ones(rec.shape[1]), np.zeros(points.shape[1])), axis=0).tolist()
                if opt.tf_summary:
                    writer.add_embedding(torch.cat((rec[0, 0:, 0:3], points[0, :, 0:3]), 0), 
                        global_step=n_iter, metadata=labels, tag="pc")

            n_iter = n_iter + 1
            batch_id = batch_id + 1

        # Output the average loss of current epoch
        avg_loss = train_loss_sum / (batch_id + 1)
        elapse = time.time() - t
        print('Epoch: %d    time: %f --- avg_loss: %f    lr: %f' % (epoch, elapse, avg_loss, scheduler.get_lr()[0]))
        if opt.tf_summary: writer.add_scalar('train/epoch_train_loss', avg_loss, epoch)
        if opt.tf_summary: writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], epoch)
        scheduler.step() # scheduler go one step

        # Save the checkpoint
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.n_epoch - 1:
            dict_name=opt.exp_name + '/epoch_'+str(epoch)+'.pth'
            torch.save({
                'model_state_dict': ae.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'last_epoch': scheduler.state_dict()['last_epoch'],
                'opt': opt
            }, dict_name)
            print('Current checkpoint saved to %s.' % (dict_name))
        print('\n')

    if opt.tf_summary: writer.close()
    print('Done!')


if __name__ == "__main__":

    option_handler = TrainOptionHandler()
    opt = option_handler.parse_options() # all options are parsed through this command
    option_handler.print_options(opt) # print out all the options
    main()
