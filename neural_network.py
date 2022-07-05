from torch import nn

import torch.nn.functional as F
from neural_fingerprint import NeuralFP
import torch
torch.manual_seed(42)

class NeuralNetwork(nn.Module):
    def __init__(self, atom_features, fp_size=2048, hidden_size=256):
        super(NeuralNetwork, self).__init__()
        self.neural_fp = NeuralFP(atom_features=atom_features,fp_size=fp_size)
        self.lin1 = nn.Linear(fp_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, 1)
        # self.lin2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, batch):
        fp = self.neural_fp(batch)
        hidden = F.relu(self.dropout(self.lin1(fp)))
        hidden = F.relu(self.dropout(self.lin2(hidden)))
        hidden = F.relu(self.dropout(self.lin3(hidden)))
        hidden = F.relu(self.dropout(self.lin4(hidden)))
        out = self.lin5(hidden)
        return out

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))


# def build_nn(num_atom_features, fp_size, hidden_sizes, optimizer, lr, wt_decay, momentum):
#     dnn = NeuralNetwork(atom_features=num_atom_features, fp_size=fp_size, hidden_sizes=hidden_sizes)
#     dnn.apply(_initialize_weights)

#     lossfn = nn.MSELoss()

#     if optimizer == 'sgd':
#         optimizer = torch.optim.SGD(dnn.parameters(), lr=lr, weight_decay=wt_decay, momentum=momentum)
#     # else:
#         # opt = optimizers.Adam(learning_rate=learning_rate)
    
#     return dnn, optimizer, lossfn

