#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Multi-object dataset based on KITTI
'''

import torch.utils.data as data
import pickle
import os
import os.path
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/'))


class KITTIMultiObjectDataset(data.Dataset):

    def __init__(self, root=dataset_path, num_points=2048, split='train'):

        with open(os.path.join(root, 'kittimulobj/kitti_mulobj_param_' + split.lower() + '_' + str(num_points) +'.pkl'), 'rb') as pickle_file:
            dict_dataset = pickle.load(pickle_file)
        with open(os.path.join(root, 'kittimulobj', dict_dataset['base_dataset'] +'.pkl'), 'rb') as pickle_file:
            self.obj_dataset = pickle.load(pickle_file)
        self.name_dict={'Pedestrian':0, 'Car':1, 'Cyclist':2, 'Van':3, 'Truck':4}
        self.npoints = num_points
        self.scene_radius = dict_dataset['scene_radius']
        self.total = dict_dataset['total']
        self.num_data_batch = dict_dataset['num_batch']
        self.batch_num_model = dict_dataset['batch_num_model']
        self.batch_num_example = dict_dataset['batch_num_example']
        self.data_list = dict_dataset['list_example']
        self.datapath = os.path.join(root, 'kittimulobj/kitti_single')
        self.max_obj_num = np.max(self.batch_num_model)
        self.obj_type_num = len(self.name_dict)
        if split.lower().find('test') >= 0: 
            self.test = True
        else: self.test = False

    def __getitem__(self, index):

        label = np.ones(self.max_obj_num, dtype=int) * -1
        point_set = []
        num_points_each = int(np.ceil(self.npoints / len(self.data_list[index]['idx'])))
        for cnt, idx_obj in enumerate(self.data_list[index]['idx']): # take out the models one-by-one
            obj_pc = np.fromfile(os.path.join(self.datapath, os.path.basename(self.obj_dataset[idx_obj]['path'])), dtype=np.float32).reshape(-1, 4) # read
            if self.test: np.random.seed(0)
            obj_pc = obj_pc[np.random.choice(obj_pc.shape[0],num_points_each), 0:3] # sample
            label[cnt] = self.name_dict[self.obj_dataset[idx_obj]['name']]
            obj_pc = self.pc_normalize(obj_pc, self.obj_dataset[idx_obj]['box3d_lidar'], self.obj_dataset[idx_obj]['name']) # normalize
            obj_pc[:,:2] += self.data_list[index]['coor'][cnt][:2] # translate
            point_set.append(obj_pc)
        point_set = np.vstack(point_set)[:self.npoints,:]

        return point_set.astype(np.float32), label

    def __len__(self):
        return self.total

    # pc: NxC, return NxC
    def pc_normalize(self, pc, bbox, label):
        pc -= bbox[:3]
        box_len = np.sqrt(bbox[3] ** 2 + bbox[4] ** 2 + bbox[5] ** 2)
        pc = pc / (box_len) * 2
        if label == 'Pedestrian' or label == 'Cyclist': 
            pc /= 2 # shrink the point sets for models with a person to better emulate a driving scene
        return pc

def draw_coor(radius):
    x = np.random.randint(radius) * 2 - (radius - 1)
    y = np.random.randint(radius) * 2 - (radius - 1)
    z = 0
    return x,y,z

# Use this main() function to generate data
def main():
    dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/kittimulobj'))
    if os.path.exists(opt.kitti_mulobj_output_file_name + '.pkl') == False:
        with open(os.path.join(dataset_path, "kitti_dbinfos_object.pkl"), 'rb') as pickle_file:
            db_obj = pickle.load(pickle_file)
        useful_obj_list = []
        max_obj_num = int(opt.kitti_mulobj_scene_radius ** 2)
        for add_obj_num in range(max_obj_num):
            useful_obj_list.append([])
        for idx_obj in range(len(db_obj)):
            for add_obj_num in range(max_obj_num):
                if db_obj[idx_obj]['num_points_in_gt'] >= int(np.ceil(opt.num_points / (add_obj_num + 1))):
                    useful_obj_list[add_obj_num].append(idx_obj)
        num_batch = len(opt.kitti_mulobj_num_example)
        list_example = []
        for idx_batch in range(num_batch):
            for idx_example in range(opt.kitti_mulobj_num_example[idx_batch]):
                print("Batch: %d,   Idx: %d" % (idx_batch, idx_example))
                coor = np.zeros((opt.kitti_mulobj_num_add_model[idx_batch], 3), dtype=float) # coordinate of the objet
                idx = np.zeros(opt.kitti_mulobj_num_add_model[idx_batch], dtype=int) # object index from the base dataset
                dr = np.zeros((opt.kitti_mulobj_num_add_model[idx_batch], 2), dtype=int) # object orientation

                # Generate object properties
                for idx_obj in range(opt.kitti_mulobj_num_add_model[idx_batch]):
                    collision = True
                    idx[idx_obj] = np.random.choice(useful_obj_list[opt.kitti_mulobj_num_add_model[idx_batch]-1]) # generate object index
                    while collision == True: # generate coordinate
                        collision = False
                        coor[idx_obj,0], coor[idx_obj,1], coor[idx_obj,2] = draw_coor(opt.kitti_mulobj_scene_radius)
                        for check_obj in range(idx_obj): # check collision between idx_obj and check_obj
                            if np.sum(np.power(coor[idx_obj,:] - coor[check_obj,:], 2)) < 4 - 1e-9:
                                collision =True
                                break

                # Generate the object index
                list_example.append({'coor':coor, 'idx':idx, 'dr':dr})

        # Save the dataset parameters
        dict_dataset = {
            'base_dataset': "kitti_dbinfos_object",
            'scene_radius': opt.kitti_mulobj_scene_radius,
            'total': sum(opt.kitti_mulobj_num_example),
            'num_batch': len(opt.kitti_mulobj_num_example),
            'batch_num_model': opt.kitti_mulobj_num_add_model,
            'batch_num_example': opt.kitti_mulobj_num_example,
            'list_example': list_example
            }
        with open(opt.kitti_mulobj_output_file_name + '.pkl', 'wb') as f:
            print(f.name)
            pickle.dump(dict_dataset, f)
        print('Dataset generation completed.')
    else: print('Dataset already exist.')


if __name__ == "__main__":

    sys.path.append(os.path.join(BASE_DIR, '..'))
    from util.option_handler import GenerateKittiMultiObjectOptionHandler
    option_handler = GenerateKittiMultiObjectOptionHandler()
    opt = option_handler.parse_options() # all options are parsed through this command
    option_handler.print_options(opt) # print out all the options
    main()