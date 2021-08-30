#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
A tool to collect the CAD models of "person", "car", "cone", "plant" from ModelNet40, and "motorbike" from ShapeNetPart
'''

import os
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from dataloaders.modelnet_loader import ModelNet40, class_extractor
from dataloaders.shapenet_part_loader import PartDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/cadmulobj')) + '/'


def main():

    label_person = 24
    label_car = 7
    label_cone = 9
    label_plant = 26
    from_shapenet_part = 'Motorbike'

    loader = ModelNet40(phase='train')
    list_person = class_extractor(label_person, loader)
    list_car = class_extractor(label_car, loader)
    list_cone = class_extractor(label_cone, loader)
    list_plant = class_extractor(label_plant, loader)
    loader = ModelNet40(phase='test')
    list_person = np.concatenate((list_person, class_extractor(label_person, loader)), axis=0)
    list_car = np.concatenate((list_car, class_extractor(label_car, loader)), axis=0)
    list_cone = np.concatenate((list_cone, class_extractor(label_cone, loader)), axis=0)
    list_plant = np.concatenate((list_plant, class_extractor(label_plant, loader)), axis=0)

    list_motorbike=[]
    loader = PartDataset(npoints=2048, classification=False, 
        class_choice=from_shapenet_part, split='trainval', normalize=True)
    for i in range(len(loader)):
        list_motorbike.append(loader[i][0].numpy())
    loader = PartDataset(npoints=2048, classification=False, 
        class_choice=from_shapenet_part, split='test', normalize=True)
    for i in range(len(loader)):
        list_motorbike.append(loader[i][0].numpy())
    list_motorbike = np.vstack(list_motorbike).reshape(-1,2048,3)

    interest_obj = {"person": list_person, "car":list_car, "cone": list_cone, "plant": list_plant, "motorbike": list_motorbike}
    np.save(os.path.join(data_path, "cad_models"), interest_obj)


if __name__ == '__main__':
    main()