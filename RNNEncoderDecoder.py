import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from util_dev import device

class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, seq_length: int, dropout:float = 0.,
                        normalization : bool = False, num_layers: int = 1):

        #input_size: feature size
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.input_size = input_size
        self.norm = normalization
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first=True)
        if normalization:
            self.bn = nn.BatchNorm1d(input_size) #along the batch_size dim
            self.layer_norm = nn.LayerNorm(hidden_size) #along the hidden_size dim

    def forward(self, input):
        ''' 
        input : (batch_size, seq_len, feat_size=1)
        output: batch_size, seq_len, hidden_size
        hidden/cell: num_layers * num_directions, batch, hidden_size
        '''
        #bz = input.size(0)
        #input = input.view(bz, seq_length, self.input_size)
        if input.ndim < 2: input.unsqueeze_(1)        
        if input.ndim < 3: input.unsqueeze_(2) #add 3rd dim to the original data
        if self.norm:
            input = self.bn(input.permute(0, 2, 1)).permute(0, 2, 1) # bn on dim=1

        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device))
        c_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device))
        output, (hidden, cell) = self.lstm(input, (h_0, c_0))

        if self.norm:
            output = self.layer_norm(output)
            hidden = self.layer_norm(hidden)
            cell = self.layer_norm(cell)

        return output, hidden, cell


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size=1, x_seq_length=6, normalization=False, dropout=0.2):
        ''' 
        output_size: output features
        '''
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.norm = normalization

        self.attn = nn.Linear(2*hidden_size + output_size, x_seq_length)
        self.attn_combine = nn.Linear(hidden_size + output_size, hidden_size)
        if normalization:
            self.bn = nn.BatchNorm1d(hidden_size)
            self.layer_norm = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        '''
        input (target): batch_size, output_feat_size(1)
        hidden: batch, hidden_size
        encoder_outputs: batch, x_seq_len, hidden_size
		output: batch, 1
        '''
        if input.ndim < 2: input.unsqueeze_(1)
        assert hidden.ndim==2
        assert cell.ndim==2

        attn_weights = F.softmax(
            self.attn(torch.cat((input, hidden, cell), 1)), dim=1).unsqueeze(1) #batch, 1, x_seq_len
        attn_applied = torch.bmm(attn_weights, encoder_outputs).squeeze(1) #batch, hidden_size

        output = torch.cat((input, attn_applied), 1)
        output = self.attn_combine(output) #batch, hidden_size
        if self.norm: 
            output = self.bn(output)        

        output = F.relu(self.dropout1(output)).unsqueeze(1) #batch, 1, hidden_size
        output, (hidden, cell) = self.lstm(output, (hidden.unsqueeze(0), cell.unsqueeze(0))) #output: batch, 1, hidden_size
        if self.norm:
            output = self.layer_norm(output)
            hidden = self.layer_norm(hidden)

        hidden = self.dropout2(hidden)
        cell = self.dropout2(cell)

        output = self.out(output.squeeze(1)) #batch, 1
        return output, hidden.squeeze(0), cell.squeeze(0), attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=device)		
