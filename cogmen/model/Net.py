import torch
import torch.nn as nn
import torch.nn.functional as F

from .GNN import GNN
from .Classifier import Classifier
from .SeqEncoder import SeqEncoder
from .GatedAttentionLayer import GattedAttention
from .mmgat import MMGAT
import cogmen

from .graph_tools import batch_graphify, late_concat

log = cogmen.utils.get_logger()

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        self.wp = args.wp
        self.wf = args.wf
        self.modalities = args.modalities
        self.n_modals = len(self.modalities)
        self.fusion_method = args.fusion_method
        self.n_speakers = args.n_speakers
        self.use_speaker = args.use_speaker
        g_dim = args.hidden_size
        h1_dim = args.hidden_size
        h2_dim = args.hidden_size
        hc_dim = args.hidden_size
        if args.gcn_conv == "mmgat":
            ic_dim = h1_dim * self.n_modals
        else:
            ic_dim = h2_dim * args.gnn_nheads * self.n_modals
        a_dim = args.dataset_embedding_dims[args.dataset]['a']
        t_dim = args.dataset_embedding_dims[args.dataset]['t']
        v_dim = args.dataset_embedding_dims[args.dataset]['v']
        
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
        }

        edge_type_to_idx = {}
        if "intra" in args.edge_type:
            for j in range(self.n_modals):
                edge_type_to_idx['-1' + str(j) + str(j)] = len(edge_type_to_idx)
                edge_type_to_idx['0' + str(j) + str(j)] = len(edge_type_to_idx)
                edge_type_to_idx['1' + str(j) + str(j)] = len(edge_type_to_idx)
        if "inter" in args.edge_type:
            for j in range(self.n_modals):
                for k in range(self.n_modals):
                    if (j != k): edge_type_to_idx['0' + str(j) + str(k)] = len(edge_type_to_idx)
        print(edge_type_to_idx)
                
        self.edge_type_to_idx = edge_type_to_idx
        self.num_relations = len(edge_type_to_idx)

        tag_size = len(dataset_label_dict[args.dataset])
        args.n_speakers = dataset_speaker_dict[args.dataset]

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device



        self.encoder = SeqEncoder(a_dim, t_dim, v_dim, g_dim, args)
        self.speaker_embedding = nn.Embedding(self.n_speakers, g_dim)
        if (args.gcn_conv == "mmgat"):
            self.gnn = MMGAT(g_dim, h1_dim, self.n_modals, self.num_relations, heads=args.gat_heads, concat=False)
        else:
            self.gnn = GNN(g_dim, h1_dim, h2_dim, self.num_relations, args)

        self.clf = Classifier(ic_dim, hc_dim, tag_size, args)


    def get_rep(self, data):

        # Encoding multimodal feature
        a = data['audio_tensor'] if 'a' in self.modalities else None
        t = data['text_tensor'] if 't' in self.modalities else None
        v = data['visual_tensor'] if 'v' in self.modalities else None

        a, t, v = self.encoder(a, t, v, data['text_len_tensor'])

        # Speaker embedding
        if self.use_speaker:
            emb = self.speaker_embedding(data['speaker_tensor'])
            a = a + emb if a != None else None
            t = t + emb if t != None else None
            v = v + emb if v != None else None

        # Graph construct
        multimodal_features = []

        if a != None:
            multimodal_features.append(a)
        if t != None:
            multimodal_features.append(t)
        if v != None:
            multimodal_features.append(v)
        node_features, node_type, edge_index, edge_type, edge_index_lengths, edge_type_lengths = \
            batch_graphify(multimodal_features, data['text_len_tensor'],
                           self.n_modals, self.wp, self.wf, 
                           self.edge_type_to_idx, self.device)
        if self.args.gcn_conv == "mmgat":
            out = self.gnn(node_features, node_type, edge_index, edge_type_lengths)
        else:
            out = self.gnn(node_features, edge_index, edge_type)

        out = late_concat(out, data['text_len_tensor'], self.n_modals)
        return out

    def forward(self, data):
        graph_out = self.get_rep(data)
        out = self.clf(graph_out, data["text_len_tensor"])

        return out
    
    def get_loss(self, data):
        graph_out = self.get_rep(data)
        loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"])
        
        return loss
        