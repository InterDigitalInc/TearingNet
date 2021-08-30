#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Reconstruction experiment
'''

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import open3d as o3d
import torch
import sys
import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from dataloaders import point_cloud_dataset_test
from models.autoencoder import PointCloudAutoencoder
from util.option_handler import TestOptionHandler
from util.mesh_writer import write_ply_mesh

import warnings
warnings.filterwarnings("ignore")

def main():
    print(torch.cuda.device_count(), "GPUs will be used for testing.")

    # Load a saved model
    if opt.checkpoint == '':
        print("Please provide the model path.")
        exit()
    else:
        checkpoint = torch.load(opt.checkpoint)
        if 'opt' in checkpoint and opt.config_from_checkpoint == True:
            checkpoint['opt'].grid_dims = opt.grid_dims
            checkpoint['opt'].xyz_chamfer_weight = opt.xyz_chamfer_weight
            ae = PointCloudAutoencoder(checkpoint['opt'])
            print("\nModel configuration loaded from checkpoint %s." % opt.checkpoint)
        else:
            ae = PointCloudAutoencoder(opt)
            checkpoint['opt'] = opt
            torch.save(checkpoint, opt.checkpoint)
            print("\nModel configuration written to checkpoint %s." % opt.checkpoint)
        ae.load_state_dict(checkpoint['model_state_dict'])
        print("Existing model %s loaded.\n" % (opt.checkpoint))
    device = torch.device("cuda:0")
    ae.to(device)
    ae.eval() # set the autoencoder to evaluation mode

    # Create a folder to write the results
    if not os.path.exists(opt.exp_name):
        os.makedirs(opt.exp_name)

    # Take care of the dataset
    _, test_dataloader = point_cloud_dataset_test(opt.dataset_name, opt.num_points, opt.batch_size, opt.test_split)
    point_cnt = opt.grid_dims[0] * opt.grid_dims[1]

    # Set the colors of all the points as 0.5 for visualization
    coloring_rec = np.ones((opt.grid_dims[0] * opt.grid_dims[1], 3)) * 0.5
    coloring = np.concatenate((coloring_rec, np.repeat(np.array([np.array(opt.gt_color)]), opt.num_points, axis=0)), axis=0)

    # Begin testing
    test_loss_sum, test_loss_sqrt_sum = 0, 0
    batch_id = 0
    print('\nTesting...')
    not_end_yet = True
    it = iter(test_dataloader)
    len_test = len(test_dataloader)

    # Iterates the testing process
    while not_end_yet == True:

        # Fetch the data
        points, _ = next(it)
        points = points.cuda()
        not_end_yet = batch_id + 1 < len_test
        if(points.size(0) < opt.batch_size): break

        # The forward pass
        with torch.no_grad():
            rec = ae(points)
        grid = rec['grid'] if 'grid' in rec else None
        if 'graph_wght' in rec:
            if opt.graph_delete_point_eps > 0: # may use another eps value
                old_eps = ae.decoder.graph_filter.graph_eps_sqr
                ae.decoder.graph_filter.graph_eps_sqr = opt.graph_delete_point_eps ** 2
                with torch.no_grad():
                    graph_wght = ae(points)['graph_wght']
                ae.decoder.graph_filter.graph_eps_sqr = old_eps
            else:
                graph_wght = rec['graph_wght']
            graph_wght = graph_wght.permute([0, 2, 3, 1]).detach().cpu().numpy()
        else: graph_wght = None
        rec = rec['rec']

        # Benchmarking the results
        test_loss, test_loss_sqrt = 0, 0
        if not(graph_wght is None) and (opt.graph_delete_point_mode >=0) and opt.graph_delete_point_eps > 0:
            idx_map = np.zeros((opt.batch_size, opt.grid_dims[0], opt.grid_dims[1]), dtype=int)
            idx = np.zeros(opt.batch_size, dtype=int)

            # When graph is used, isolated points are removed
            for i in range(opt.grid_dims[0]):
                for j in range(opt.grid_dims[1]):
                    degree = np.sum(graph_wght[:, i, j, :] > opt.graph_thres, axis=1)
                    tag = (degree <= opt.graph_delete_point_mode) # tag the point to be deleted
                    idx_map[:, i, j][tag == True] = -1
                    idx_map[:, i, j][tag == False] = idx[tag == False]
                    idx[tag == False] = idx[tag == False] + 1
            for b in range(opt.batch_size):
                pc_cur = rec[b,idx_map[b].reshape(opt.grid_dims[0] * opt.grid_dims[1])>=0].unsqueeze(0)
                test_loss += ae.xyz_loss(points[b].unsqueeze(0), pc_cur, xyz_loss_type=2)
                test_loss_sqrt += ae.xyz_loss(points[b].unsqueeze(0), pc_cur, xyz_loss_type=0)
            test_loss /= opt.batch_size
            test_loss_sqrt /= opt.batch_size
            test_loss_sum = test_loss_sum + test_loss
            test_loss_sqrt_sum = test_loss_sqrt_sum + test_loss_sqrt
        else:
            test_loss = ae.xyz_loss(points, rec, xyz_loss_type=2)
            test_loss_sqrt = ae.xyz_loss(points, rec, xyz_loss_type=0)
            test_loss_sum += test_loss.item()
            test_loss_sqrt_sum += test_loss_sqrt.item()

        if batch_id % opt.print_freq == 0:
            print('    batch_no: %d/%d, ch_dist: %f, ch^2_dist: %f' % 
                (batch_id, len_test, test_loss_sqrt.item(), test_loss.item()))

        # Write down the first point cloud
        if opt.pc_write_freq > 0 and batch_id % opt.pc_write_freq == 0:
            rec_o3d = o3d.geometry.PointCloud()
            rec = rec[0].data.cpu()
            rec_o3d.colors = o3d.utility.Vector3dVector(coloring[:rec.shape[0],:])
            rec_o3d.points = o3d.utility.Vector3dVector(rec)
            file_rec = os.path.join(opt.exp_name, str(batch_id) + "_rec.ply")
            o3d.io.write_point_cloud(file_rec, rec_o3d)

            gt_o3d = o3d.geometry.PointCloud() # write the ground-truth
            gt_o3d.points = o3d.utility.Vector3dVector(points[0].data.cpu())
            gt_o3d.colors = o3d.utility.Vector3dVector(coloring[rec.shape[0]:,:])
            file_gt = os.path.join(opt.exp_name, str(batch_id) + "_gt.ply")
            o3d.io.write_point_cloud(file_gt, gt_o3d)

            if not(grid is None): # write the torn grid
                grid = torch.cat((grid[0].contiguous().data.cpu(), torch.zeros(point_cnt, 1)), 1)
                grid_o3d = o3d.geometry.PointCloud()
                grid_o3d.points = o3d.utility.Vector3dVector(grid)
                grid_o3d.colors = o3d.utility.Vector3dVector(coloring_rec)
                file_grid = os.path.join(opt.exp_name, str(batch_id) + "_grid.ply")
                o3d.io.write_point_cloud(file_grid, grid_o3d)

            if not(graph_wght is None) and opt.write_mesh:
                output_path = os.path.join(opt.exp_name, str(batch_id) + "_rec_mesh.ply") # write the reconstructed mesh
                write_ply_mesh(rec.view(opt.grid_dims[0], opt.grid_dims[1], 3).numpy(), 
                    coloring_rec.reshape(opt.grid_dims[0], opt.grid_dims[1], 3), opt.graph_edge_color, 
                    output_path, thres=opt.graph_thres, delete_point_mode=opt.graph_delete_point_mode, 
                    weights=graph_wght[0], point_color_as_index=opt.point_color_as_index)

                output_path = os.path.join(opt.exp_name, str(batch_id) + "_grid_mesh.ply") # write the grid mesh
                write_ply_mesh(grid.view(opt.grid_dims[0], opt.grid_dims[1], 3).numpy(), 
                    coloring_rec.reshape(opt.grid_dims[0], opt.grid_dims[1], 3), opt.graph_edge_color, 
                    output_path, thres=opt.graph_thres, delete_point_mode=-1, 
                    weights=graph_wght[0], point_color_as_index=opt.point_color_as_index)

        batch_id = batch_id + 1

    # Log the test results
    avg_loss = test_loss_sum / (batch_id + 1)
    avg_loss_sqrt = test_loss_sqrt_sum / (batch_id + 1)
    print('avg_ch_dist: %f    avg_ch^2_dist: %f' % (avg_loss_sqrt, avg_loss))

    print('\nDone!\n')

if __name__ == "__main__":

    option_handler = TestOptionHandler()
    opt = option_handler.parse_options() # all options are parsed through this command
    option_handler.print_options(opt) # print out all the options
    main()
