import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv, GATConv, RGATConv, GATv2Conv


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, args):
        super(GNN, self).__init__()
        
        self.num_relations = num_relations
        if args.gcn_conv == "rgcn":
            self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)
        elif args.gcn_conv == "gat":
            self.conv1 = GATConv(g_dim, h1_dim, concat=False, heads=args.gat_heads)
        elif args.gcn_conv == "rgat":
            self.conv1 = RGATConv(g_dim, h1_dim, self.num_relations, concat=False, heads=args.gat_heads)
        elif args.gcn_conv == "gatv2":
            self.conv1 = GATv2Conv(g_dim, h1_dim, edge_dim=1, concat=False, heads=args.gat_heads)

        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=args.gnn_nheads, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * args.gnn_nheads)

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = nn.functional.leaky_relu(self.bn(self.conv2(x, edge_index)))

        return x
