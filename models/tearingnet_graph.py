#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
TearingNet with graph filtering
'''

import torch
import torch.nn as nn

from .foldingnet import FoldingNetVanilla
from .tearingnet import TearingNetBasic


class GraphFilter(nn.Module):

    def __init__(self, grid_dims, graph_r, graph_eps, graph_lam):
        super(GraphFilter, self).__init__()
        self.grid_dims = grid_dims
        self.graph_r = graph_r
        self.graph_eps_sqr = graph_eps * graph_eps
        self.graph_lam = graph_lam

    def forward(self, grid, pc):
        
        # Data preparation
        bs_cur = pc.shape[0]
        grid_exp = grid.contiguous().view(bs_cur, self.grid_dims[0], self.grid_dims[1], 2) # batch_size X dim0 X dim1 X 2
        pc_exp = pc.contiguous().view(bs_cur, self.grid_dims[0], self.grid_dims[1], 3) # batch_size X dim0 X dim1 X 3
        graph_feature = torch.cat((grid_exp, pc_exp), dim=3).permute([0, 3, 1, 2])

        # Compute the graph weights
        wght_hori = graph_feature[:,:,:-1,:] - graph_feature[:,:,1:,:] # horizontal weights
        wght_vert = graph_feature[:,:,:,:-1] - graph_feature[:,:,:,1:] # vertical weights
        wght_hori = torch.exp(-torch.sum(wght_hori * wght_hori, dim=1) / self.graph_eps_sqr) # Gaussian weight
        wght_vert = torch.exp(-torch.sum(wght_vert * wght_vert, dim=1) / self.graph_eps_sqr)
        wght_hori = (wght_hori > self.graph_r) * wght_hori
        wght_vert = (wght_vert > self.graph_r) * wght_vert
        wght_lft = torch.cat((torch.zeros([bs_cur, 1, self.grid_dims[1]]).cuda(), wght_hori), 1) # add left
        wght_rgh = torch.cat((wght_hori, torch.zeros([bs_cur, 1, self.grid_dims[1]]).cuda()), 1) # add right
        wght_top = torch.cat((torch.zeros([bs_cur, self.grid_dims[0], 1]).cuda(), wght_vert), 2) # add top
        wght_bot = torch.cat((wght_vert, torch.zeros([bs_cur, self.grid_dims[0], 1]).cuda()), 2) # add bottom
        wght_all = torch.cat((wght_lft.unsqueeze(1), wght_rgh.unsqueeze(1), wght_top.unsqueeze(1), wght_bot.unsqueeze(1)), 1)

        # Perform the actural graph filtering: x = (I - \lambda L) * x
        wght_hori = wght_hori.unsqueeze(1).expand(-1, 3, -1, -1) # dimension expansion
        wght_vert = wght_vert.unsqueeze(1).expand(-1, 3, -1, -1)
        pc = pc.permute([0, 2, 1]).contiguous().view(bs_cur, 3, self.grid_dims[0], self.grid_dims[1])
        pc_filt = \
            torch.cat((torch.zeros([bs_cur, 3, 1, self.grid_dims[1]]).cuda(), pc[:,:,:-1,:] * wght_hori), 2) + \
            torch.cat((pc[:,:,1:,:] * wght_hori, torch.zeros([bs_cur, 3, 1, self.grid_dims[1]]).cuda()), 2) + \
            torch.cat((torch.zeros([bs_cur, 3, self.grid_dims[0], 1]).cuda(), pc[:,:,:,:-1] * wght_vert), 3) + \
            torch.cat((pc[:,:,:,1:] * wght_vert, torch.zeros([bs_cur, 3, self.grid_dims[0], 1]).cuda()), 3) # left, right, top, bottom

        pc_filt = pc + self.graph_lam * (pc_filt - torch.sum(wght_all,dim=1).unsqueeze(1).expand(-1, 3, -1, -1) * pc) # equivalent to ( I - \lambda L) * x
        pc_filt = pc_filt.view(bs_cur, 3, -1).permute([0, 2, 1])
        return pc_filt, wght_all


class TearingNetGraphModel(nn.Module):

    @staticmethod
    def add_options(parser, isTrain = True):

        # General optional(s)
        parser.add_argument('--grid_dims', type=int, nargs='+', help='Grid dimensions.')

        # Options related to the Folding Network
        parser.add_argument('--folding1_dims', type=int, nargs='+', default=[514, 512, 512, 3], help='Dimensions of the first folding module.')
        parser.add_argument('--folding2_dims', type=int, nargs='+', default=[515, 512, 512, 3], help='Dimensions of the second folding module.')

        # Options related to the Tearing Network
        parser.add_argument('--tearing1_dims', type=int, nargs='+', default=[523, 256, 128, 64], help='Dimensions of the first tearing module.')
        parser.add_argument('--tearing2_dims', type=int, nargs='+', default=[587, 256, 128, 2], help='Dimensions of the second tearing module.')
        parser.add_argument('--tearing_conv_kernel_size', type=int, default=1, help='Kernel size of the convolutional layers in the Tearing Network, 1 implies MLP.')

        # Options related to graph construction
        parser.add_argument('--graph_r', type=float, default=1e-12, help='Parameter r for the r-neighborhood graph.')
        parser.add_argument('--graph_eps', type=float, default=0.02, help='Parameter epsilon for the graph (bandwidth parameter).')
        parser.add_argument('--graph_lam', type=float, default=0.5, help='Parameter lambda for the graph filter.')

        return parser

    def __init__(self, opt):
        super(TearingNetGraphModel, self).__init__()

        # Initialize the regular 2D grid
        range_x = torch.linspace(-1.0, 1.0, opt.grid_dims[0])
        range_y = torch.linspace(-1.0, 1.0, opt.grid_dims[1])
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # Initialize the Folding Network and the Tearing Network
        self.folding = FoldingNetVanilla(opt.folding1_dims, opt.folding2_dims)
        self.tearing = TearingNetBasic(opt.tearing1_dims, opt.tearing2_dims, opt.grid_dims, opt.tearing_conv_kernel_size)
        self.graph_filter = GraphFilter(opt.grid_dims, opt.graph_r, opt.graph_eps, opt.graph_lam)

    def forward(self, cw):

        grid0 = self.grid.cuda().unsqueeze(0).expand(cw.shape[0], -1, -1) # batch_size X point_num X 2
        pc0 = self.folding(cw, grid0) # Folding Network
        grid1 = self.tearing(cw, grid0, pc0) # Tearing Network
        pc1 = self.folding(cw, grid1) # Folding Network
        pc2, graph_wght = self.graph_filter(grid1, pc1) # Graph Filtering
        return pc0, pc1, pc2, grid1, graph_wght
