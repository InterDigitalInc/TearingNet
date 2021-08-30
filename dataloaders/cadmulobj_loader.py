#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Multi-object dataset based on CAD models
'''

import torch.utils.data as data
import os
import sys
import os.path
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/'))


class CADMultiObjectDataset(data.Dataset):

    def __init__(self, root=dataset_path, num_points=2048, split=None, normalize=True):
        from . import cad_models_loader
        self.npoints = num_points
        self.normalize = normalize
        dict_dataset = np.load(os.path.join(root, 'cadmulobj/cad_mulobj_param_' + split.lower() + '.npy'), allow_pickle=True).item()
        self.scene_radius = dict_dataset['scene_radius']
        self.total = dict_dataset['total']
        self.augmentation = dict_dataset['augmentation']
        self.num_data_batch = dict_dataset['num_batch']
        self.batch_num_model = dict_dataset['batch_num_model']
        self.batch_num_example = dict_dataset['batch_num_example']
        self.data_list = dict_dataset['list_example']
        self.max_obj_num = np.max(self.batch_num_model)
        self.base_dataset = cad_models_loader.CADModelsDataset(num_points=2048, normalize=True)
        self.obj_type_num = self.base_dataset.obj_type_num

    def __getitem__(self, index):

        index_new = index
        label = np.ones(self.max_obj_num, dtype=int) * -1
        point_set = []
        num_points_each = int(np.ceil(self.npoints / len(self.data_list[index_new]['idx'])))

        for cnt, idx_obj in enumerate(self.data_list[index_new]['idx']): # take out the models one-by-one
            obj_pc = self.pc_normalize(self.base_dataset[idx_obj][0][:num_points_each, :])
            label[cnt] = self.base_dataset[idx_obj][1]
            trans = self.data_list[index_new]['coor'][cnt].copy()
            if self.augmentation == True: # rotation augmentation if needed
                obj_pc = obj_pc @ self.gen_rotation_matrix(self.data_list[index_new]['dr'][cnt][0], 1)
            trans[1] = trans[1] - np.min(obj_pc[:,1])
            point_set.append(obj_pc + trans)

        point_set = np.vstack(point_set)[:self.npoints,:]
        return point_set.astype(np.float32), label

    def __len__(self):
        return self.total

    # pc: NxC, return NxC
    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    # Generate rotation matrix, theta is in terms of degree
    def gen_rotation_matrix(self, theta, rotation_axis):
        all_axis = np.eye(3)
        rotation_theta = np.deg2rad(theta)
        rotation_axis = all_axis[rotation_axis,:]
        sin_xyz = np.sin(rotation_theta)*rotation_axis
        R = np.cos(rotation_theta)*np.eye(3)
        R[0,1] = -sin_xyz[2]
        R[0,2] = sin_xyz[1]
        R[1,0] = sin_xyz[2]
        R[1,2] = -sin_xyz[0]
        R[2,0] = -sin_xyz[1]
        R[2,1] = sin_xyz[0]
        R = R + (1-np.cos(rotation_theta))*np.dot(np.expand_dims(rotation_axis, axis = 1), np.expand_dims(rotation_axis, axis = 0))
        return R


def draw_coor(radius):
    x = np.random.randint(radius) * 2 - (radius - 1)
    y = np.random.randint(radius) * 2 - (radius - 1)
    z = 0
    return x,y,z


# Use this main() function to generate data
def main():

    if os.path.exists(opt.cad_mulobj_output_file_name + '.npy') == False:
        num_batch = len(opt.cad_mulobj_num_example)
        list_example = []
        for idx_batch in range(num_batch):
            for idx_example in range(opt.cad_mulobj_num_example[idx_batch]):
                print("Batch: %d,   Idx: %d" % (idx_batch, idx_example))
                coor = np.zeros((opt.cad_mulobj_num_add_model[idx_batch], 3), dtype=float) # coordinate of the objet
                idx = np.zeros(opt.cad_mulobj_num_add_model[idx_batch], dtype=int) # object index from the base dataset
                dr = np.zeros((opt.cad_mulobj_num_add_model[idx_batch], 2), dtype=int) # object orientation

                # Generate object properties
                for idx_obj in range(opt.cad_mulobj_num_add_model[idx_batch]):
                    collision = True
                    if opt.augmentation:
                        dr[idx_obj, 0], dr[idx_obj, 1] = np.random.randint(0,360), np.random.randint(0,3) # generate direction
                    else: dr[idx_obj, 0], dr[idx_obj, 1] = 0, 0
                    idx[idx_obj] = np.random.randint(opt.cad_mulobj_num_ava_model) # generate object index
                    while collision == True: # generate coordinate
                        collision = False
                        coor[idx_obj,2], coor[idx_obj,0], coor[idx_obj,1] = draw_coor(opt.cad_mulobj_scene_radius)
                        for check_obj in range(idx_obj): # check collision between idx_obj and check_obj
                            if np.sum(np.power(coor[idx_obj,:] - coor[check_obj,:], 2)) < 4 - 1e-9:
                                collision =True
                                break

                # Generate the object index
                list_example.append({'coor':coor, 'idx':idx, 'dr':dr})

        # Save the dataset parameters
        dict_dataset = {
            'scene_radius': opt.cad_mulobj_scene_radius,
            'total': sum(opt.cad_mulobj_num_example),
            'augmentation': opt.augmentation,
            'ava_model_idx': opt.cad_mulobj_num_ava_model,
            'num_batch': len(opt.cad_mulobj_num_example),
            'batch_num_model': opt.cad_mulobj_num_add_model,
            'batch_num_example': opt.cad_mulobj_num_example,
            'list_example': list_example
            }
        np.save(opt.cad_mulobj_output_file_name, dict_dataset)
        print('Dataset %s generation completed.' % opt.cad_mulobj_output_file_name)

if __name__ == "__main__":

    sys.path.append(os.path.join(BASE_DIR, '..'))
    from util.option_handler import GenerateCADMultiObjectOptionHandler
    option_handler = GenerateCADMultiObjectOptionHandler()
    opt = option_handler.parse_options() # all options are parsed through this command
    option_handler.print_options(opt) # print out all the options
    main()