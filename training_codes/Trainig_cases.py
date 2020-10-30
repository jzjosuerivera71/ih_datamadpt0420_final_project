import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Preprocesing Module

from modules.p_preprocesing import m_preprocesing as pp     
from modules.p_model import p_model as mod


#DATA

#dataframe = pd.read_csv('global_data.csv')
#dataframe = dataframe.iloc[:,1:2].values

#SCALER   

sc = MinMaxScaler()
training_data = sc.fit_transform(dataframe)

#SHIFTING   

seq_length = 4
X, y = pp.shifting(training_data, seq_length)

# SEPARATE DATA

train_X, train_Y, test_X, test_Y = pp.separate_data(X, y)


#PARAMETERS        
input_dim = 1;
hidden_layer_size = 2
num_layers = 1
output_dim = 1

num_of_epochs = 2000
display_step  = 100
learning_rate = 0.01

model = mod.LSTM(input_dim, output_dim, hidden_layer_size, num_layers)

# METRIC AND OPTIMIZATION

MSE = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#TRAINING

for epoch in range(num_of_epochs):

    outputs = model(train_X)
    loss = MSE(outputs, train_Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % display_step == 0:
        print('Epoch: {:d}, Loss: {:.4f}'.format(epoch, loss.item()))

torch.save(model.state_dict(), 'LSTM.pkl')

###########################

USA_training = pd.read_csv('data/USA_training.csv')

cases = USA_training[['cases']].values

sc = MinMaxScaler()
seq_length = 4


# Cases

training_data_cases = sc.fit_transform(cases)

cases_X, cases_y = pp.shifting(training_data_cases, seq_length)

# SEPARATE DATA

C_train_X, C_train_Y, C_test_X, C_test_Y = pp.separate_data(cases_X, cases_y)



# MODEL

model = mod.LSTM(input_dim, output_dim, hidden_layer_size, num_layers)



# METRIC AND OPTIMIZATION

MSE = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(num_of_epochs):

    outputs = model(C_train_X)
    loss = MSE(outputs, C_train_Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % display_step == 0:
        print('Epoch: {:d}, Loss: {:.4f}'.format(epoch, loss.item()))

torch.save(model.state_dict(), 'LSTM_cases.pkl')



