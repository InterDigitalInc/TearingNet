
import torch
import numpy as np
from . import cadmulobj_loader, kittimulobj_loader


# Build the dataset for trainging accordingly
def point_cloud_dataset_train(dataset_name, num_points, batch_size, train_split='train', val_batch_size=1, num_workers=8):
    train_dataset = None
    train_dataloader = None
    val_dataset = None
    val_dataloader = None
    if dataset_name.lower() == 'cad_mulobj': # CAD model multiple-object dataset
        train_dataset = cadmulobj_loader.CADMultiObjectDataset(num_points=2048, split=train_split, normalize=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    elif dataset_name.lower() == 'kitti_mulobj': # KITTI multiple-object dataset
        train_dataset = kittimulobj_loader.KITTIMultiObjectDataset(num_points=2048, split=train_split)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataset, train_dataloader, val_dataset, val_dataloader


# Build the dataset for testing accordingly
def point_cloud_dataset_test(dataset_name, num_points, batch_size, test_split='test', test_class=None, num_workers=8):
    test_dataset = None
    if dataset_name.lower() == 'cad_mulobj': # Our multiple-object dataset
        test_dataset = cadmulobj_loader.CADMultiObjectDataset(num_points=2048, split=test_split, normalize=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    elif dataset_name.lower() == 'kitti_mulobj': # Our multiple-object dataset
        test_dataset = kittimulobj_loader.KITTIMultiObjectDataset(num_points=2048, split=test_split)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_dataset, test_dataloader


# Function to generate rotation matrix
def gen_rotation_matrix(theta=-1, rotation_axis=-1):
    all_theta = [0, 90, 180, 270]
    all_axis = np.eye(3)

    if theta == -1:
        theta = all_theta[np.random.randint(0, 4)]
    elif theta == -2:
        theta = np.random.rand() * 360
    elif theta == -3:
        theta == (np.random.rand() - 0.5) * 90
    else: theta = all_theta[theta]
    rotation_theta = np.deg2rad(theta)

    if rotation_axis == -1:
        rotation_axis = all_axis[np.random.randint(0, 3), :]
    else: rotation_axis = all_axis[rotation_axis,:]
    sin_xyz = np.sin(rotation_theta) * rotation_axis

    R = np.cos(rotation_theta) * np.eye(3)

    R[0,1] = -sin_xyz[2]
    R[0,2] = sin_xyz[1]
    R[1,0] = sin_xyz[2]
    R[1,2] = -sin_xyz[0]
    R[2,0] = -sin_xyz[1]
    R[2,1] = sin_xyz[0]
    R = R + (1 - np.cos(rotation_theta)) * np.dot(np.expand_dims(rotation_axis, axis=1), np.expand_dims(rotation_axis, axis=0))
    R = torch.from_numpy(R)
    return R.float()