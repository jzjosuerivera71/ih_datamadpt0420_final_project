import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import datetime
from sklearn.preprocessing import MinMaxScaler

# ORGANIZATION OF THE DATA

def Prep_cases(dataframe):
    #
    lst_state = list(set(dataframe['state']))
    timestamps = list(set(dataframe['date']))
    #
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps]
    dates.sort()
    sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
    #
    # INITIALIZATION
    sum_cases = [[lst_state[i],[]] for i in range(len(lst_state))]
    #
    for i in range(len(lst_state)):
        state = dataframe[dataframe['state']== lst_state[i]]
        for j in range(len(sorteddates)):
            lst = list(state[state['date']== sorteddates[j]]['cases'])
            sum_deaths[i][1].append(sum(lst))
    #
    dict_cases = {k: v for k, v in sum_cases}
    data = pd.DataFrame(dict_cases)
    data['date']= pd.Series(sorteddates)
    return data

def Prep_deaths(dataframe):
    #
    lst_state = list(set(dataframe['state']))
    timestamps = list(set(dataframe['date']))
    #
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps]
    dates.sort()
    sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
    #
    # INITIALIZATION
    sum_deaths = [[lst_state[i],[]] for i in range(len(lst_state))]
    #
    for i in range(len(lst_state)):
        state = dataframe[dataframe['state']== lst_state[i]]
        for j in range(len(sorteddates)):
            lst = list(state[state['date']== sorteddates[j]]['deaths'])
            sum_deaths[i][1].append(sum(lst))
    #
    dict_deaths = {k: v for k, v in sum_deaths}
    data = pd.DataFrame(dict_deaths)
    data['date']= pd.Series(sorteddates)
    return data



# PREPARATION FOR THE MODEL

def shifting(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:i+seq_length]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

def separate_data(X, y):
    train_size = int(len(y) * 0.9)
    test_size = len(y) - train_size

    train_X = torch.Tensor(X[0:train_size])
    train_Y = torch.Tensor(y[0:train_size])

    test_X = torch.Tensor(X[train_size:])
    test_Y = torch.Tensor(y[train_size:])

    return train_X, train_Y, test_X, test_Y

# SCALER
sc = MinMaxScaler()

#Parameters
input_dim = 1;
hidden_layer_size = 2
num_layers = 1
output_dim = 1
seq_length = 4

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
    
def Size(state_data):
	training_set = state_data.to_numpy()
	training_set = training_set[0]  # REDUCCION DE DIMENSION
	training_set = np.delete(training_set, [0])
	training_set = training_set.reshape(len(training_set), 1)

	training_size = int(len(training_set)* 0.80) # Trainig Size
	return training_size


def Prediction(state_data,model_path):
	training_set = state_data.to_numpy()
	training_set = training_set[0]  # REDUCCION DE DIMENSION
	training_set = np.delete(training_set, [0])
	training_set = training_set.reshape(len(training_set), 1)
	#
	# MODEL
	model = LSTM(input_dim, output_dim, hidden_layer_size, num_layers)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	#
	# Training Data
	training_set = sc.fit_transform(training_set)
	#
	# ESTO ES PARA ESTABLECER METRICAS DE EVALUACION
	X, y = shifting(training_set, seq_length)
	dataX = torch.Tensor(X)
	dataY = torch.Tensor(y)
	train_X, train_Y, test_X, test_Y = separate_data(X, y)
	#
	train_predict = model(dataX)
	data_predict = train_predict.data.numpy()
	data_predict = sc.inverse_transform(data_predict)
	return data_predict

def Actual(state_data):
	training_set = state_data.to_numpy()
	training_set = training_set[0]  # REDUCCION DE DIMENSION
	training_set = np.delete(training_set, [0])
	training_set = training_set.reshape(len(training_set), 1)
	#
	# Training Data
	training_set = sc.fit_transform(training_set)
	X, y = shifting(training_set, seq_length)
	dataY = torch.Tensor(y)
	actual_data= dataY.data.numpy()
	actual_data = sc.inverse_transform(actual_data)	
	return actual_data    
