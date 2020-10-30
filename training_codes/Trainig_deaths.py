import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Preprocesing Module

from modules.p_preprocesing import m_preprocesing as pp     
from modules.p_model import p_model as mod


#DATA
USA_training = pd.read_csv('data/USA_training.csv')

deaths = USA_training[['deaths']].values


#SCALER
sc = MinMaxScaler()
seq_length = 4


# Cases

training_data = sc.fit_transform(deaths)

X, y = pp.shifting(training_data, seq_length)

# SEPARATE DATA

train_X, train_Y, test_X, test_Y = pp.separate_data(X,y)

#PARAMETERS        
input_dim = 1;
hidden_layer_size = 2
num_layers = 1
output_dim = 1

num_of_epochs = 2000
display_step  = 100
learning_rate = 0.01

# MODEL

model = mod.LSTM(input_dim, output_dim, hidden_layer_size, num_layers)



# METRIC AND OPTIMIZATION

MSE = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(num_of_epochs):

    outputs = model(train_X)
    loss = MSE(outputs,train_Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % display_step == 0:
        print('Epoch: {:d}, Loss: {:.4f}'.format(epoch, loss.item()))

torch.save(model.state_dict(), 'LSTM_deaths.pkl')


