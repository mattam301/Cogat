from comet_ml import Experiment, Optimizer

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cogmen
from cogmen.model.graph_tools import batch_graphify, late_concat
log = cogmen.utils.get_logger()

import time

if __name__ == '__main__':
    class Obj(object):
        pass

    data = cogmen.utils.load_pkl('data/iemocap/data_iemocap.pkl')
    args = Obj()
    setattr(args, 'batch_size', 32)
    setattr(args, 'dataset', 'iemocap')
    setattr(args, 'dataset_embedding_dims', {"iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 100 + 768,
            "tv": 768 + 512,
            "av": 612,
            "atv": 100 + 768 + 512,
        }})
    
    setattr(args, 'modalities', 'atv')
    data = cogmen.Dataset(data["train"], args)

    text_tensor = torch.arange(0, 10, 1).view(2, -1, 1)
    audio_tensor = torch.arange(10, 20, 1).view(2, -1, 1)
    visual_tensor = torch.arange(20, 30, 1).view(2, -1, 1)
    edge_type_to_idx = {}
        
    for j in range(3):
        edge_type_to_idx['-1' + str(j) + str(j)] = len(edge_type_to_idx)
        edge_type_to_idx['0' + str(j) + str(j)] = len(edge_type_to_idx)
        edge_type_to_idx['1' + str(j) + str(j)] = len(edge_type_to_idx)
    
    for j in range(3):
        for k in range(3):
            if (j != k): edge_type_to_idx['0' + str(j) + str(k)] = len(edge_type_to_idx)


    m= [text_tensor, audio_tensor, visual_tensor]
    length = torch.tensor([5, 5])

    node_features, node_type, edge_index, edge_type, edge_index_lengths, edge_type_lengths = \
        batch_graphify(m, length, 3, -1, -1, edge_type_to_idx, 'cuda')
    
    print(node_features)
    print(node_type)
    print(edge_index)
    print(edge_type_lengths)



    



