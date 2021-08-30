#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Basic FoldingNet model
'''

import torch
import torch.nn as nn
from .pointnet import PointwiseMLP


class FoldingNetVanilla(nn.Module):

    def __init__(self, folding1_dims, folding2_dims):
        super(FoldingNetVanilla, self).__init__()

        # The folding decoder
        self.fold1 = PointwiseMLP(folding1_dims, doLastRelu=False)
        if folding2_dims[0] > 0:
            self.fold2 = PointwiseMLP(folding2_dims, doLastRelu=False)
        else: self.fold2 = None

    def forward(self, cw, grid, **kwargs):

        cw_exp = cw.unsqueeze(1).expand(-1, grid.shape[1], -1) # batch_size X point_num X code_length

        # 1st folding
        in1 = torch.cat((grid, cw_exp), 2) # batch_size X point_num X (code_length + 3)
        out1 = self.fold1(in1) # batch_size X point_num X 3

        # 2nd folding
        if not(self.fold2 is None):
            in2 = torch.cat((out1, cw_exp), 2) # batch_size X point_num X (code_length + 4)
            out2 = self.fold2(in2) # batch_size X point_num X 3
            return out2
        else: return out1


class FoldingNetVanillaModel(nn.Module):

    @staticmethod
    def add_options(parser, isTrain = True):

        # Some optionals
        parser.add_argument('--grid_dims', type=int, nargs='+', help='Grid dimensions.')
        parser.add_argument('--folding1_dims', type=int, nargs='+', default=[514, 512, 512, 3], help='Dimensions of the first folding module.')
        parser.add_argument('--folding2_dims', type=int, nargs='+', default=[515, 512, 512, 3], help='Dimensions of the second folding module.')
        return parser

    def __init__(self, opt):
        super(FoldingNetVanillaModel, self).__init__()

        # Initialize the 2D grid
        range_x = torch.linspace(-1.0, 1.0, opt.grid_dims[0])
        range_y = torch.linspace(-1.0, 1.0, opt.grid_dims[1])
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # Initialize the folding module
        self.folding1 = FoldingNetVanilla(opt.folding1_dims, opt.folding2_dims)

    def forward(self, cw):

        grid = self.grid.cuda().unsqueeze(0).expand(cw.shape[0], -1, -1) # batch_size X point_num X 2
        pc = self.folding1(cw, grid)
        return pc