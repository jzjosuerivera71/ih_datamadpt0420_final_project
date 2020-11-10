#https://colah.github.io/posts/2015-08-Understanding-LSTMs/

import torch
import torch.nn as nn

#PARAMETERS        
input_dim = 1;
hidden_layer_size = 2
num_layers = 1
output_dim = 1


# Esto no es necesario aqui

num_of_epochs = 2000
display_step  = 100
learning_rate = 0.01

#MODEL    

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, num_layers):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_layer_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, input):
        hidden_state = torch.zeros(self.num_layers, input.size(0), self.hidden_layer_size)
        cell_state = torch.zeros(self.num_layers, input.size(0), self.hidden_layer_size)

        output, (hidden_state, cell_state) = self.lstm(input, (hidden_state, cell_state))
        out = hidden_state.view(-1, 2)

        out = self.fc(out)

        return out



