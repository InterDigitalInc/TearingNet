#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Mesh writer
'''

import numpy as np

def dist(i, j):
    return (i[0]-j[0]) ** 2 + (i[1]-j[1]) ** 2 + (i[2]-j[2]) ** 2

def write_ply_mesh(points, color, edge_color, filepath, thres=0, delete_point_mode=-1, weights=None, point_color_as_index=False, thres_edge=1e3):
    f = open(filepath, "w")

    if weights is None:
        weights = np.zeros((points.shape[0], points.shape[1], 4), dtype=int)
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                if i > 0: weights[i,j,0] = 1e6 # left available
                if i < points.shape[0] - 1: weights[i,j,1] = 1e6 # right available
                if j > 0: weights[i,j,2] = 1e6 # top available
                if j < points.shape[1] - 1: weights[i,j,3] = 1e6 # bottom available

    # Initialize the map for drawing points
    idx = 0
    idx_map = np.zeros((points.shape[0], points.shape[1]), dtype=int)
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            degree  = sum(weights[i,j,:] > thres)
            if degree <= delete_point_mode: # found a point to be removed
                idx_map[i,j] = -1
            else:
                idx_map[i,j] = idx
                idx += 1

    # Calculate total number of edges
    edge_total = 0
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            if idx_map[i,j] >= 0:
                if weights[i,j,0] > thres and idx_map[i-1,j] >= 0 and dist(points[i,j,:], points[i-1,j,:]) <= thres_edge: edge_total += 1 # left
                if weights[i,j,1] > thres and idx_map[i+1,j] >= 0 and dist(points[i,j,:], points[i+1,j,:]) <= thres_edge: edge_total += 1 # right
                if weights[i,j,2] > thres and idx_map[i,j-1] >= 0 and dist(points[i,j,:], points[i,j-1,:]) <= thres_edge: edge_total += 1 # top
                if weights[i,j,3] > thres and idx_map[i,j+1] >= 0 and dist(points[i,j,:], points[i,j+1,:]) <= thres_edge: edge_total += 1 # bottom

    # Calculate total number of faces
    face_total = 0
    for i in range(points.shape[0] - 1):
        for j in range(points.shape[1] - 1):
            if (weights[i,j,1] > thres) and (weights[i,j,3] > thres) and (weights[i+1,j+1,0]> thres) and (weights[i+1,j+1,2]> thres):
                face_total +=1

    # Write header
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex " + str(idx) + "\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("element face " + str(face_total * 4) + "\n")
    f.write("property list uchar int vertex_index\n")
    f.write("element edge " + str(edge_total) + "\n")
    f.write("property int vertex1\n")
    f.write("property int vertex2\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    # Write points
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            if idx_map[i,j] >= 0:
                f.write(str(points[i,j,0]) + " " + str(points[i,j,1]) + " " + str(points[i,j,2]) + " ")
                if point_color_as_index == True: f.write(str(i) + " " + str(j) + " 0\n")
                else: f.write(str(int(color[i,j,0] * 255)) + " " + str(int(color[i,j,1] * 255)) + " " + str(int(color[i,j,2] * 255)) + "\n")

    # Write faces
    for i in range(points.shape[0] - 1):
        for j in range(points.shape[1] - 1):
            if (weights[i,j,1] > thres) and (weights[i,j,3] > thres) and (weights[i+1,j+1,0]> thres) and (weights[i+1,j+1,2]> thres):
                # f.write("4 " + str(idx_map[i+1,j]) + " " + str(idx_map[i+1,j+1]) + " " + str(idx_map[i,j+1]) + " " + str(idx_map[i,j]) + "\n")
                f.write("3 " + str(idx_map[i+1,j]) + " " + str(idx_map[i+1,j+1]) + " " + str(idx_map[i,j+1]) + "\n")
                f.write("3 " +  str(idx_map[i,j+1]) + " " + str(idx_map[i,j]) + " " + str(idx_map[i+1,j]) + "\n")
                f.write("3 " + str(idx_map[i,j+1]) + " " + str(idx_map[i+1,j+1]) + " " + str(idx_map[i+1,j]) + "\n")
                f.write("3 " + str(idx_map[i+1,j]) + " " + str(idx_map[i,j]) + " " + str(idx_map[i,j+1]) + "\n")

    # Write meshes
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            if idx_map[i,j] >= 0:
                if weights[i,j,0] > thres and idx_map[i-1,j] >= 0 and dist(points[i,j,:], points[i-1,j,:]) <= thres_edge: # left
                    f.write(str(idx_map[i,j]) + " " + str(idx_map[i-1,j]) + " " + str(int(edge_color[0] * 255)) + 
                        " " + str(int(edge_color[1] * 255)) + " " + str(int(edge_color[2] * 255)) + "\n")
                if weights[i,j,1] > thres and idx_map[i+1,j] >= 0 and dist(points[i,j,:], points[i+1,j,:]) <= thres_edge: # right
                    f.write(str(idx_map[i,j]) + " " + str(idx_map[i+1,j]) + " " + str(int(edge_color[0] * 255)) + 
                        " " + str(int(edge_color[1] * 255)) + " " + str(int(edge_color[2] * 255)) + "\n")
                if weights[i,j,2] > thres and idx_map[i,j-1] >= 0 and dist(points[i,j,:], points[i,j-1,:]) <= thres_edge: # top
                    f.write(str(idx_map[i,j]) + " " + str(idx_map[i,j-1]) + " " + str(int(edge_color[0] * 255)) + 
                        " " + str(int(edge_color[1] * 255)) + " " + str(int(edge_color[2] * 255)) + "\n")
                if weights[i,j,3] > thres and idx_map[i,j+1] >= 0 and dist(points[i,j,:], points[i,j+1,:]) <= thres_edge: # bottom
                    f.write(str(idx_map[i,j]) + " " + str(idx_map[i,j+1]) + " " + str(int(edge_color[0] * 255)) + 
                        " " + str(int(edge_color[1] * 255)) + " " + str(int(edge_color[2] * 255)) + "\n")
    f.close()