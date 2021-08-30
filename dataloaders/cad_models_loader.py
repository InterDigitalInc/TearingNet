'''
Dataloader for CAD models of "person", "car", "cone", "plant" from ModelNet40, and "motorbike" from ShapeNetPart
'''

import torch.utils.data as data
import os
import os.path
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/'))


class CADModelsDataset(data.Dataset):

    def create_cad_models_dataset_pickle(self, root):
        dict_dataset = np.load(os.path.join(root, 'cadmulobj/cad_models.npy'), allow_pickle=True).item()
        data_person = dict_dataset['person'] # 0. person
        data_car = dict_dataset['car'] # 1. car
        data_cone = dict_dataset['cone'] # 2. cone
        data_plant = dict_dataset['plant'] # 3. plant
        data_motorbike = dict_dataset['motorbike'] # 4. motorbike

        label_person = np.ones(data_person.shape[0], dtype=int) * 0
        label_car = np.ones(data_car.shape[0], dtype=int) * 1
        label_cone = np.ones(data_cone.shape[0], dtype=int) * 2
        label_plant = np.ones(data_plant.shape[0], dtype=int) * 3
        label_motorbike = np.ones(data_motorbike.shape[0], dtype=int) * 4
        self.data = np.concatenate((data_person, data_car, data_cone, data_plant, data_motorbike), axis=0)
        self.label = np.concatenate((label_person, label_car, label_cone, label_plant, label_motorbike), axis=0)
        self.total = self.data.shape[0]
        self.obj_type_num = 5

    def __init__(self, root=dataset_path, num_points=2048, normalize=True):
        self.npoints = num_points
        self.normalize = normalize
        self.create_cad_models_dataset_pickle(root)

    def __getitem__(self, index):
        point_set = self.data[index, 0 : self.npoints, :]
        label = self.label[index]
        if self.normalize:
            point_set = self.pc_normalize(point_set)
        return point_set, label

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
