#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Object counting experiment

author: jpang
created: Apr 27, 2020 (Mon) 13:55 EST
'''

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch
import torch.nn.parallel
import sys
import os
import numpy as np
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from dataloaders import point_cloud_dataset_test
from models.autoencoder import PointCloudAutoencoder
from util.option_handler import CountingOptionHandler
import time

import warnings
warnings.filterwarnings("ignore")

def main():
    print(torch.cuda.device_count(), "GPUs will be used for object counting.")

    # Build up an autoencoder
    t = time.time()

    # Load a saved model
    if opt.checkpoint == '':
        print("Please provide the model path.")
        exit()
    else:
        path_checkpoints = [os.path.join(opt.checkpoint, f) for f in os.listdir(opt.checkpoint) 
            if os.path.isfile(os.path.join(opt.checkpoint, f)) and f[-1]=='h']
        if len(path_checkpoints) > 1:
            path_checkpoints.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Take care of the dataset
    test_dataset, test_dataloader = point_cloud_dataset_test(opt.dataset_name, 
        opt.num_points, opt.batch_size, opt.count_split)

    # Create folder to save results
    if not os.path.exists(opt.exp_name):
        os.makedirs(opt.exp_name)
    device = torch.device("cuda:0")

    # Go through each checkpoint in the folder
    max_avg_mae, min_avg_mae = 0, 1e6
    max_min_mae, min_min_mae = 0, 1e6
    for cnt, str_checkpoint in enumerate(path_checkpoints):

        if len(path_checkpoints) > 1:
            epoch_cur = int(''.join(filter(str.isdigit, str_checkpoint[str_checkpoint.find('epoch_'):])))
            if (opt.epoch_interval[0] < 0) and (len(path_checkpoints) - cnt > opt.epoch_interval[1]): continue
            if (opt.epoch_interval[0] >= 0) and (epoch_cur >= opt.epoch_interval[0] and epoch_cur <= opt.epoch_interval[1])== False: continue
        else: epoch_cur = -1
        checkpoint = torch.load(str_checkpoint)

        # Load the model, may use model configuration from checkpoint
        if 'opt' in checkpoint and opt.config_from_checkpoint == True:
            checkpoint['opt'].grid_dims = opt.grid_dims
            ae = PointCloudAutoencoder(checkpoint['opt'])
            print("\nModel configuration loaded from checkpoint %s." % str_checkpoint)
        else:
            ae = PointCloudAutoencoder(opt)
            checkpoint['opt'] = opt
            torch.save(checkpoint, str_checkpoint)
            print("\nModel configuration written to checkpoint %s." % str_checkpoint)
        ae.load_state_dict(checkpoint['model_state_dict'])
        print("\nExisting model %s loaded." % (str_checkpoint))
        ae.to(device)
        ae.eval() # set the autoencoder to evaluation mode

        # Compute the codewords
        print('Computing codewords..')
        batch_cw = []
        batch_labels = []
        torch.cuda.empty_cache()
        len_test = len(test_dataloader)
        with torch.no_grad():
            for batch_id, data in enumerate(test_dataloader):
                points, labels = data
                if(points.size(0) < opt.batch_size): break
                points = points.cuda()
                with torch.no_grad():
                    rec = ae(points)
                cw = rec['cw']
                batch_cw.append(cw.detach().cpu().numpy())
                if (opt.dataset_name.lower() != 'torus_orig') and (opt.dataset_name.lower() != 'torus'):
                    labels = torch.sum(labels>=0, dim=1)
                else: labels = labels + 1
                batch_labels.append(labels.detach().cpu().numpy())
                if batch_id % opt.print_freq == 0:
                    elapse = time.time() - t
                    print('    batch_no: %d/%d, time: %f' % (batch_id, len_test, elapse))
        batch_cw = np.vstack(batch_cw)
        batch_labels = np.vstack(batch_labels).reshape((-1,)).astype(int)

        # Divide the folds for experiments
        data_cnt_total = len(batch_labels)
        data_fold_idx = np.zeros(data_cnt_total, dtype=int)
        if (opt.dataset_name.lower() != 'torus_orig') and (opt.dataset_name.lower() != 'torus'):
            obj_cnt = np.zeros(test_dataset.max_obj_num, dtype=int)
        else: obj_cnt = np.zeros(3, dtype=int)
        for cur_obj in range(data_cnt_total):
            obj_cnt[batch_labels[cur_obj]-1] += 1
            data_fold_idx[cur_obj] = obj_cnt[batch_labels[cur_obj]-1]
        obj_cnt = (obj_cnt / opt.count_fold).astype(int)
        for cur_obj in range(data_cnt_total):
            data_fold_idx[cur_obj] = int((data_fold_idx[cur_obj]-1) / (obj_cnt[batch_labels[cur_obj]-1] + 1e-10))

        # Perform testing for each fold
        mae = np.zeros(opt.count_fold, dtype=float)
        for cur_fold in range(opt.count_fold):
            data_train = batch_cw[data_fold_idx==cur_fold, :]
            label_train = batch_labels[data_fold_idx==cur_fold]
            data_test = batch_cw[data_fold_idx!=cur_fold, :]
            label_test = batch_labels[data_fold_idx!=cur_fold]

            # Train and test the classifier
            classifier = SVC(gamma='scale', C=opt.svm_params[0])
            classifier.fit(data_train, label_train)
            pred = classifier.predict(data_test)
            mae[cur_fold] = np.mean(abs(label_test-pred))
            print('    fold: {}    '.format(cur_fold+1) +  'mae: {:.4f}    '.format(mae[cur_fold]) )
        print('avg_mae: {:.4f}    '.format(np.mean(mae)))
        mae_idx = np.argmin(mae)
        print('min_mae: {:.4f}    '.format(mae[mae_idx]) + 'min_mae_idx: {}    '.format(mae_idx+1))

        if np.mean(mae) > max_avg_mae:
            max_avg_mae = np.mean(mae)
            max_avg_mae_epoch = epoch_cur
        if np.mean(mae) < min_avg_mae:
            min_avg_mae = np.mean(mae)
            min_avg_mae_epoch = epoch_cur
        if mae[mae_idx] > max_min_mae:
            max_min_mae = mae[mae_idx]
            max_min_mae_epoch = epoch_cur
        if mae[mae_idx] < min_min_mae:
            min_min_mae = mae[mae_idx]
            min_min_mae_epoch = epoch_cur

    print('\nmax_avg_mae: {:.4f},'.format(max_avg_mae) + ' max_avg_mae_epoch: {:d}'.format(max_avg_mae_epoch))
    print('min_avg_mae: {:.4f},'.format(min_avg_mae) + ' min_avg_mae_epoch: {:d}'.format(min_avg_mae_epoch))
    print('max_min_mae: {:.4f},'.format(max_min_mae) + ' max_min_mae_epoch: {:d}'.format(max_min_mae_epoch))
    print('min_min_mae: {:.4f},'.format(min_min_mae) + ' min_min_mae_epoch: {:d}'.format(min_min_mae_epoch))

    print('\nDone!')

if __name__ == "__main__":

    option_handler = CountingOptionHandler()
    opt = option_handler.parse_options() # all options are parsed through this command
    option_handler.print_options(opt) # print out all the options
    main()
