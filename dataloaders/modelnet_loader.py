'''
Modified based on https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/dataloaders/modelnet40_loader.py
'''

import os
import numpy as np
from torch.utils.data import Dataset
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/modelnet40')) + '/'

class ModelNet40(Dataset):
    
    def load_h5(self, file_name):
        f = h5py.File(file_name)
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    def __init__(self, data_path=data_path, num_points=2048, transform=None,
                 phase='train'):
        self.data_path = os.path.join(data_path, 'modelnet40_ply_hdf5_2048')
        self.num_points = num_points
        self.num_classes = 40
        self.transform = transform

        # store data
        shape_name_file = os.path.join(self.data_path, 'shape_names.txt')
        self.shape_names = [line.rstrip() for line in open(shape_name_file)]
        self.coordinates = []
        self.labels = []
        try:
            files = os.path.join(self.data_path, '{}_files.txt'.format(phase))
            files = [line.rstrip() for line in open(files)]
            for index, file in enumerate(files):
                file_name = file.split('/')[-1]
                files[index] = os.path.join(self.data_path, file_name)
        except FileNotFoundError:
            raise ValueError('Unknown phase or invalid data path.')
        for file in files:
            current_data, current_label = self.load_h5(file)
            current_data = current_data[:, 0:self.num_points, :]
            self.coordinates.append(current_data)
            self.labels.append(current_label)
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)

    def __len__(self):
        return self.coordinates.shape[0]

    def __getitem__(self, index):
        # coord = np.transpose(self.coordinates[index])  # 3 * N
        coord = self.coordinates[index]
        label = self.labels[index]
        data = (coord,)
        # transform coordinates
        if self.transform is not None:
            transformed, matrix, mask = self.transform(coord)
            data += (transformed, matrix, mask)
        data += (label,)
        return data

def class_extractor(label, loader):
    list_obj=[]
    for i in range(len(loader)):
        data, label_cur = loader.__getitem__(i)
        if label_cur == label:
            list_obj.append(data)
    list_obj = np.vstack(list_obj).reshape(-1,2048,3)
    return list_obj


if __name__ == '__main__':
    main()