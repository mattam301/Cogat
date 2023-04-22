import torch
import numpy as np
import time

def batch_graphify(multimodal_feature, lengths, n_modals, wp, wf, edge_type_to_idx, device, intra=True, inter=True):
    node_features, node_type, edge_index, edge_type, edge_index_lengths = [], [], [], [], []
    edge_type_lengths = [0] * len(edge_type_to_idx)

    sum_length = 0
    total_length = lengths.sum().item()
    batch_size = lengths.size(0)

    for k, feature in enumerate(multimodal_feature):
        for j in range(batch_size):
            cur_len = lengths[j].item()
            node_features.append(feature[j,:cur_len])
            node_type.extend([k] * cur_len)
    
    for j in range(batch_size):
        cur_len = lengths[j].item()
        
        perms = edge_perms(cur_len, wp, wf, n_modals, total_length, intra, inter)
        edge_index_lengths.append(len(perms))
        
        for item in perms:
            vertices = item[0]
            neighbor = item[1]
            edge_index.append(torch.tensor([vertices + sum_length, neighbor + sum_length]))
            
            if vertices % total_length > neighbor % total_length:
                temporal_type = 1
            elif vertices % total_length < neighbor % total_length:
                temporal_type = -1
            else:
                temporal_type = 0
            edge_type.append(edge_type_to_idx[str(temporal_type) 
                                              + str(node_type[vertices + sum_length])
                                              + str(node_type[neighbor + sum_length])])

        sum_length += cur_len

    indices = np.argsort(edge_type)

    node_features = torch.cat(node_features, dim=0).to(device)
    node_type = torch.tensor(node_type).long().to(device)
    sorted_edge_type, sorted_edge_index = [], []

    for j in indices:
        sorted_edge_type.append(edge_type[j])
        sorted_edge_index.append(edge_index[j])
        edge_type_lengths[edge_type[j]] += 1
    
    edge_index = torch.stack(sorted_edge_index).t().contiguous().to(device)  # [2, E]
    edge_type = torch.tensor(sorted_edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]
    edge_type_lengths = torch.Tensor(edge_type_lengths).long().to(device) # [num_edge_types]
    return node_features, node_type, edge_index, edge_type, edge_index_lengths, edge_type_lengths

def edge_perms(length, window_past, window_future, n_modals, total_lengths, intra=True, inter=True):
    
    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[: min(length, j + window_future)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past) :]
        else:
            eff_array = array[
                max(0, j - window_past) : min(length, j + window_future)
            ]
        perms = set()

        
        for k in range(n_modals):
            node_index = j + k * total_lengths
            if intra == True:
                for item in eff_array:
                    perms.add((node_index, item + k * total_lengths))
            else:
                perms.add((node_index, node_index))
            if inter == True:
                for l in range(n_modals):
                    if l != k:
                        perms.add((node_index, j + l * total_lengths))

        all_perms = all_perms.union(perms)
            
    return list(all_perms)


def late_concat(nodes_feature, lengths, n_modals):
    batch_size = lengths.size(0)
    sum_length = lengths.sum().item()
    feature = []
    for j in range(n_modals):
        feature.append(nodes_feature[j * sum_length : (j + 1) * sum_length])
        
    feature = torch.cat(feature, dim=1)

    return feature

def joint_concat(multimodal_feature, lengths, dim):
    batch_size = lengths.size(0)
    max_len = torch.max(lengths).item()
    n_modals = len(multimodal_feature)

    feature_out = torch.zeros((batch_size, max_len, dim))

    for j in range(batch_size):
        tmp = []
        cur_len = lengths[j].item()

        for feature in multimodal_feature:
            tmp.append(feature[j])
        batch = torch.cat(tmp, dim=1)

        feature_out[j, : , :] = batch
    
    return feature_out

if __name__ == '__main__':
    pass