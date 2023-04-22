import torch
import torch.nn as nn

class GattedAttention(nn.Module):
    
    def __init__(self, g_dim, h_dim, args):
        super(GattedAttention, self).__init__()
        self.g_dim = g_dim
        self.h_dim = h_dim
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

        self.transform_t = nn.Linear(g_dim, h_dim, bias=True)
        self.transform_a = nn.Linear(g_dim, h_dim, bias=True)
        self.transform_v = nn.Linear(g_dim, h_dim, bias=True)

        self.transform_av = nn.Linear(g_dim*3, 1)
        self.transform_at = nn.Linear(g_dim*3, 1)
        self.transform_tv = nn.Linear(g_dim*3, 1)
    
    def forward(self, a, t, v):
        a = self.dropout_a(a) if a != None else None
        t = self.dropout_t(t) if t != None else None
        v = self.dropout_a(v) if v != None else None

        ha = torch.tanh(self.transform_a(a)) if a != None else None
        ht = torch.tanh(self.transform_t(t)) if t != None else None
        hv = torch.tanh(self.transform_v(v)) if v != None else None

        if ha != None and hv != None:
            z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a*v], dim=-1)))
            h_av = z_av * ha + (1 - z_av)*hv
            if ht == None:
                return h_av
        

        if ha != None and ht != None:
            z_at = torch.sigmoid(self.transform_at(torch.cat([a, t, a*t], dim=-1)))
            h_at = z_at * ha + (1 - z_at)*ht
            if hv == None:
                return h_at
        

        if ht != None and hv != None:
            z_tv = torch.sigmoid(self.transform_tv(torch.cat([t, v, t*v], dim=-1)))
            h_tv = z_tv*ht + (1 - z_tv)*hv
            if ha == None:
                return h_tv
        
        return torch.cat([h_av, h_at, h_tv], dim=-1)
        

    

        