import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class SeqTransfomer(nn.Module):
    def __init__(self, input_size,h_dim, args):
        super(SeqTransfomer, self).__init__()

        self.input_size = input_size

        self.nhead = 2
        for h in range(7, 15):
            if self.input_size % h == 0:
                self.nhead = h
                break

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,
                                                   nhead=self.nhead,
                                                   dropout=args.drop_rate,
                                                   batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, args.seqcontext_nlayer)

        self.transformer_out = torch.nn.Linear(
                input_size, h_dim, bias=True
            )
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.transformer_out(x)
        return x


class FC_with_PE(nn.Module):
    def __init__(self, input_size, h_dim, args):
        super(FC_with_PE, self).__init__()

        self.input_size = input_size
        self.hidden_dim = h_dim
        self.args = args

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.fc(x)
        return x

class LSTM_Layer(nn.Module):
    def __init__(self, input_size, h_dim, args):
        super(LSTM_Layer, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = h_dim

        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_dim // 2,
                            dropout=args.drop_rate,
                            bidirectional=True,
                            num_layers=args.seqcontext_nlayer,
                            batch_first=True)
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)

        packed_out, (_, _) = self.lstm(packed, None)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        return out

<<<<<<< HEAD
class Bert_layer(torch.nn.Module):
    def __init__(self, t_dim, hidden_dim, args):
        super(Bert_layer, self).__init__()
        self.t_dim = t_dim
        self.hidden_dim = hidden_dim
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_text):
        encoded_input = self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        outputs = self.text_encoder(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
        return outputs.last_hidden_state
=======
>>>>>>> parent of 3083208 (test import BERT)

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # Added to support odd d_model
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)