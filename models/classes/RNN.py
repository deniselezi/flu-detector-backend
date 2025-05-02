import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, output_size=28, hidden_size=[200, 100], dropout=None):
        super(GRU, self).__init__()
        self.num_layers = len(hidden_size)
        self.hidden_sizes = hidden_size
        
        self.gru_layers = nn.ModuleList()
        for i in range(self.num_layers):
                in_size = input_size if i == 0 else hidden_size[i - 1]
                self.gru_layers.append(nn.GRU(in_size, hidden_size[i], num_layers=1, batch_first=True))

        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc = nn.Linear(hidden_size[-1], output_size)
    
    def forward(self, x):
        h = [torch.zeros(1, x.size(0), hidden_size).to(x.device) for hidden_size in self.hidden_sizes]
        
        out = x
        for i, gru_layer in enumerate(self.gru_layers):
            out, h[i] = gru_layer(out, h[i])
            if i == 0 and self.dropout is not None:
                out = self.dropout(out)

        out = self.fc(out[:, -1, :])
        return out
