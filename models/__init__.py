#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Initialization and simple utilities
'''

import importlib

def get_model_class(model_name):

    # enumerate the model files being used here
    model_file_list = ["pointnet", "autoencoder", "foldingnet", "tearingnet", "tearingnetgraph"]
 
    model_class = None
    for filename in model_file_list:

        # Retrieve the model class
        modellib = importlib.import_module("models." + filename)
        class_name = model_name + "Model"
        for name, cls in modellib.__dict__.items():
            if name.lower() == class_name.lower():
                model_class = cls
                break
        if model_class is not None:
            break
    
    if model_class is None:
        print("The specified model [{}] not found.".format(model_name))
        exit(0)

    return model_class