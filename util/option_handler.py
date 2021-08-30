#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Option handler
'''

import argparse
import models

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BasicOptionHandler():

    def add_options(self, parser):
        parser.add_argument('--exp_name', type=str, default='experiment_name', help='Name of the experiment, folders are created by this name.')
        parser.add_argument('--phase', type=str, default='train', help='Indicate current phase, train or test')
        parser.add_argument('--encoder', type=str, default='pointnetvanilla', help='Choice of the encoder.')
        parser.add_argument('--decoder', type=str, default='foldingnetvanilla', help='Choice of the decoder.')
        parser.add_argument('--checkpoint', type=str, default='', help='Restore from a indicated checkpoint.')
        parser.add_argument('--load_weight_only', type=str2bool, nargs='?', const=True, default=False, help='Load the model weight only when restoring from a checkpoint.')
        parser.add_argument('--xyz_loss_type', type=int, default=0, help='Choose the loss type for point cloud.')
        parser.add_argument('--xyz_chamfer_weight', type=float, default=1, help='Balance the two terms of the Chamfer distance.')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size when loding the dataset.')
        parser.add_argument('--num_points', type=int, default=2048, help='Input point set size')
        parser.add_argument('--dataset_name', default='', help='Dataset name')
        parser.add_argument('--config_from_checkpoint', type=str2bool, nargs='?', const=True, default=False, help='Load the model configuration form checkpoint.')
        parser.add_argument('--tf_summary', type=str2bool, nargs='?', const=True, default=True, help='Whether to use tensorboard for log.')
        return parser

    def parse_options(self):

        # Initialize parser with basic options
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.add_options(parser)

        # Get the basic options
        opt, _ = parser.parse_known_args()

        # Train or test
        if opt.phase.lower() == 'train':
            self.isTrain = True
        elif opt.phase.lower() == 'test' or opt.phase.lower() == 'counting':
            self.isTrain = False
        elif (opt.phase.lower() == 'gen_cadmultiobj') or (opt.phase.lower() == 'kitti_data') or (opt.phase.lower() == 'gen_kittimulobj'):
            self.parser = parser
            self.isTrain = False
            return opt
        else:
            print("The phase [{}] does not exist.".format(opt.phase))
            exit(0)
        opt.isTrain = self.isTrain   # train or test

        # Add options to the parser according to the chosen models
        encoder_option_setter = models.get_model_class(opt.encoder).add_options
        parser = encoder_option_setter(parser, self.isTrain)
        decoder_option_setter = models.get_model_class(opt.decoder).add_options
        parser = decoder_option_setter(parser, self.isTrain)

        self.parser = parser
        opt, _ = parser.parse_known_args()
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        # For k, v in sorted(vars(opt).items()):
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

class TrainOptionHandler(BasicOptionHandler):

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--n_epoch', type=int, default=100, help='Number of epoch to train.')
        parser.add_argument('--print_freq', type=int, default=20, help='Frequency of displaying training results.')
        parser.add_argument('--pc_write_freq', type=int, default=1000, help='Frequency of writing down the point cloud during training, <= 0 means do not write.')
        parser.add_argument('--save_epoch_freq', type=int, default=2, help='Frequency of saving the trained model.')
        parser.add_argument('--lr', type=float, nargs='+', default=[0.0002], help='Learning rate and its related parameters.')
        parser.add_argument('--train_split', type=str, default='train', help='The split of the dataset for training.')
        parser.add_argument('--lr_module', type=float, nargs='+', help='Specify the learning rate of the modules.')
        parser.add_argument('--optim_args', type=float, nargs='+', default=[0.9, 0.999, 0], help='Parameters of the ADAM optimizer.')
        parser.add_argument('--augmentation', type=str2bool, nargs='?', const=True, default=False, help='Apply data augmentation or not.')
        parser.add_argument('--augmentation_theta', type=int, default=-1, help='Choice of the rotation angle from [0,90,180,270].')
        parser.add_argument('--augmentation_rotation_axis', type=int, default=-1, help='Choice of the rotation axis from np.eye(3).')
        parser.add_argument('--augmentation_flip_axis', type=int, default=-1, help='The axis around which to flip for augmentation.')
        parser.add_argument('--augmentation_min_scale', type=float, default=1, help='Minimum of the scaling factor for augmentation.')
        parser.add_argument('--augmentation_max_scale', type=float, default=1, help='Maximum of the scaling factor for augmentation.')

        return parser

class TestOptionHandler(BasicOptionHandler):

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--test_split', type=str, default='test', help='Specify the split being used.')
        parser.add_argument('--print_freq', type=int, default=20, help='Frequency of displaying results during testing.')
        parser.add_argument('--pc_write_freq', type=int, default=1000, help='Frequency of writing down the point cloud during testing.')        
        parser.add_argument('--gt_color', type=float, nargs='+', default=[0, 0, 0], help='Color of the ground-truth point cloud.')
        parser.add_argument('--write_mesh', type=str2bool, nargs='?', const=True, default=False, help='Write down meshes or not.')
        parser.add_argument('--graph_thres', type=float, default=-1, help='Threshold of the graph edges.')
        parser.add_argument('--graph_edge_color', type=float, nargs='+', default=[0.5, 0.5, 0.5], help='Color of the graph edges.')
        parser.add_argument('--graph_delete_point_mode', type=int, default=-1, help='Mode of removing points: -1: no removal; 0: remove those without edge; 1: remove those with one edge.')        
        parser.add_argument('--graph_delete_point_eps', type=float, default=-1, help='The epsilon to used when evaluating the point-deleted-version point cloud.')
        parser.add_argument('--thres_edge', type=float, default=3, help='Threshold of the length of the edge to be written.')
        parser.add_argument('--point_color_as_index', type=str2bool, nargs='?', const=True, default=False, help='Whether regard the point cloud color as point index.')
        return parser

class CountingOptionHandler(BasicOptionHandler):

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--count_split', type=str, default='test', help='The split of the dataset for object counting.')
        parser.add_argument('--count_fold', type=int, default=4, help='Number of fold for doing the object counting experiment.')
        parser.add_argument('--epoch_interval', type=int, nargs='+', default=[-1, 10], help='Range of the epoch to do object counting.')
        parser.add_argument('--print_freq', type=int, default=20, help='Frequency of displaying object counting results.')
        parser.add_argument('--svm_params', type=float, nargs='+', default=[1e3], help='SVM parameters.')
        return parser

class GenerateCADMultiObjectOptionHandler(BasicOptionHandler):

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--cad_mulobj_num_add_model', type=int, nargs='+', default=[2, 3, 4], help='A list. Numbers of 3D models to add into the scene.')
        parser.add_argument('--cad_mulobj_num_example', type=int, nargs='+', default=[1000, 1000, 1000], help='A list. Numbers of examples corresponds to each type of scene.')
        parser.add_argument('--cad_mulobj_num_ava_model', type=int, default=1000, help='Total number of available models in the original dataset.')
        parser.add_argument('--cad_mulobj_scene_radius', type=float, default=5, help='Radius of the generated scene.')
        parser.add_argument('--cad_mulobj_output_file_name', type=str, required=True, help='Path and file name of the output file parametrizing the dataset.')
        parser.add_argument('--augmentation', type=str2bool, nargs='?', const=True, default=False, help='Apply data augmentation or not.')
        return parser

class GenerateKittiMultiObjectOptionHandler(BasicOptionHandler):

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--kitti_mulobj_num_add_model', type=int, nargs='+', default=[2, 3, 4], help='A list. Numbers of 3D models to add into the scene.')
        parser.add_argument('--kitti_mulobj_num_example', type=int, nargs='+', default=[1000, 1000, 1000], help='A list. Numbers of examples corresponds to each type of scene.')
        parser.add_argument('--kitti_mulobj_scene_radius', type=float, default=5, help='Radius of the generated scene.')
        parser.add_argument('--kitti_mulobj_output_file_name', type=str, required=True, help='Path and the filename of the output file parametrizing the dataset.')
        return parser