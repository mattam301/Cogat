import torch
import torch.nn as nn

from .Transfomer import SeqTransfomer, FC_with_PE, LSTM_Layer, Bert_layer
class SeqEncoder(nn.Module):
    def __init__(self, a_dim, t_dim, v_dim, h_dim, args):
        super(SeqEncoder, self).__init__()
        self.hidden_dim = h_dim
        self.device = args.device
        self.rnn = args.rnn
       
        self.a_dim = a_dim
        self.t_dim = t_dim
        self.v_dim = v_dim

        print("modalities's dim: " + a_dim, t_dim, v_dim)

        ## Audio Encoding
        self.audio_encoder = nn.Sequential(nn.Linear(self.a_dim, self.hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_dim, self.hidden_dim),
                                           nn.ReLU())

        # Text Encoding
        if (self.rnn == "transformer"):
            self.text_encoder = SeqTransfomer(self.t_dim, self.hidden_dim, args)
        elif (self.rnn == "ffn"):
            self.text_encoder = nn.Sequential(nn.Linear(self.t_dim, self.hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_dim, self.hidden_dim),
                                           nn.ReLU())
        elif (self.rnn == "lstm"):
            self.text_encoder = LSTM_Layer(self.t_dim, self.hidden_dim, args)
        # elif (self.rnn == "bert"):
        #     self.text_encoder = Bert_layer(self.t_dim, self.hidden_dim, args)
            
            

        # Visual Encoding
        self.vision_encoder = nn.Sequential(nn.Linear(self.v_dim, self.hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_dim, self.hidden_dim),
                                           nn.ReLU())


    def forward(self, a, t, v, lengths):
        if a == None:
            a_out = None
        else:
            a_out = self.audio_encoder(a)

        if t == None:
            t_out = None
        else:
            if (self.rnn == "lstm"):
                t_out = self.text_encoder(t, lengths)
            else:
                t_out = self.text_encoder(t)

        
        if v == None:
            v_out = None
        else:
            v_out = self.vision_encoder(v)

        return t_out, a_out, v_out
    