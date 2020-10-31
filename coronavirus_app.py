

#https://docs.streamlit.io/en/stable/api.html

#
#https://github.com/whiteboxml/teaching-ironhack-dataptmad-2004/blob/master/week_10/introduction_to_matplotlib_and_seaborn/introduction_to_matplotlib_and_seaborn.ipynb

#DEPENDENCIES

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
#%matplotlib inline


#MODULES
from p_modules import m_modules as pp
from p_analysis import m_analysis as an


#########################################################3

#INTRODUCTION

st.title('LSTM neural network for timeseries evolution of Coronavirus.')


st.header('Introduction.')

st.subheader('Problem description:')
st.write('The main idea of the project is to train several recurrent neural networks to be able to predict the evolution of Covid19 as accurately as possible. This will help us to make better decisions about population control.')


st.subheader('Original Model.')
st.write('From this differntial equation:')
st.latex( r'''  \partial_t f = f(t)(N - f(t))  ''')     
st.write('we can deduce the solution that we use to do the sigmoidal regretion. This regretion is the most used for this type of analisis')     
st.latex(r''' f(t) = \frac {N}{1 + e^{-k(t - t_0)}} ''')
st.write('This model is very useful but there is some limitations, especialy in the fluctuations of the growth')

########################################################

# METODOLOGY

st.header('Metodology.')


st.subheader('LSTM neural Network Model')
st.write('This option is better to predict the evolution of the disease because the prediction is adjusted according to the previous data. Additionally this model is easy to train.')
#image = ??????
#st.image(image, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', output_format='auto', **kwargs)
# st.header("A cat")
st.image("data/LSTM.png", use_column_width=True)

st.subheader('Pipeline dependencies:')
st.write('- numpy')
st.write('- pandas')
st.write('- pytorch')
st.write('- matplotlib')
st.write('- datetime')
st.write('- scipy')
st.write('- sklearn')
st.write('- streamlit')

st.subheader('Pipeline componets:') 
st.write('- 3 modules (preprocesing, analisis, models)')
st.write('- 1 LSTM class')
st.write('- 2 train neural networks')
st.write('- 3 analisis functions')
st.write('- 4 preprocesing functions')

st.subheader('Training Scheme:')
st.write('The original dataset grouped the accumulated cases and deaths by county. To do the training, the totals had to be added by date.') 
st.graphviz_chart('''
    digraph {
        dataset -> totals
        totals -> cases
        totals -> deaths
        cases -> training
        deaths -> training
        training -> models
    }
''')       
     

st.subheader('Data transformation:')
st.write('To make the predictions, the same had to be done, but for each state.The result was two datasets which are used in the demonstration.')  
st.graphviz_chart('''
    digraph {
        original -> states_data
        states_data -> LSTM
        LSTM -> Predictions
    }
''')     

       
st.subheader('Pytorch LSTM class ')
st.write("This is the most importand part of the pipeline.")

code = '''class LSTM(nn.Module):
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

        		return out'''
     
st.code(code, language='python')


##########################################################

#DEMO
st.header('Demo of the LSTM pipeline')

# Data
cases = pd.read_csv('data/acumulated_cases.csv')
deaths = pd.read_csv('data/acumulated_deaths.csv')
states = tuple(cases.columns)


# Dropdown
modes = ('Cases','Deaths')
option_1 = st.selectbox(
     'What do you want to know?',
     modes)

st.write('You selected:', option_1)


option_2 = st.selectbox(
     'Which state are you interested?',
     states)

st.write('You selected:', option_2)


#PATHS

cases_model_path = 'models/LSTM_cases.pkl'
deaths_model_path = 'models/LSTM_deaths.pkl'

#DATA TO PREDICT

state_case = cases[[option_2]].T
deaths_case = deaths[[option_2]].T

	
# Lo que falta

#plt.ylabel('# of Confirmed Cases')
#plt.xlabel('# of Days From First Case')


###############################


# CASES
cases_size = pp.Size(state_case)
cases_predict = pp.Prediction(state_case,cases_model_path)
actual_cases = pp.Actual(state_case)

#DEATHS
deaths_size = pp.Size(deaths_case)
deaths_predict = pp.Prediction(deaths_case,deaths_model_path)
actual_deaths = pp.Actual(deaths_case)


#################################################

# ANALISIS

# Sigmoidal function

data_cases = cases[option_2]
data_deaths = deaths[option_2]

def log_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))
    
def selection_x(option_1):
	if option_1 == 'Cases':
		x_data = range(len(cases.index))
		return x_data
	else:
		x_data = range(len(deaths.index))
		return x_data
    
def selection_y(option_1,data_cases,data_deaths):
	if option_1 == 'Cases':
		#x_data = range(len(cases.index))
		y_data = data_cases
		return y_data
	else:
		#x_data = range(len(cases.index))
		y_data = data_deaths
		return y_data
			
# Fit the curve
x_data = selection_x(option_1)
y_data = selection_y(option_1,data_cases,data_deaths)
popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0,0,0],np.inf), maxfev=50000)
estimated_k, estimated_x_0, ymax= popt

# Plot the fitted curve
k = estimated_k
x_0 = estimated_x_0
y_fitted = log_curve(range(0,230), k, x_0, ymax)


# Calculations

#growth = an.growth_factor(cases_predict)
#an.growth_ratio(confirmed)


###################################################################

# PLOT

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
if  option_1 == 'Cases':
	#fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
	ax[0].set_title('LSTM')
	ax[0].plot(cases_predict, c='r', label='LSTM acumulated cases predictions')
	ax[0].plot(actual_cases, c='b', label='Actual Data')
	ax[0].axvline(x=cases_size, linestyle='--')
	ax[0].legend(loc='upper left')
	#
	ax[1].set_title('Sigmoidal')
	ax[1].plot(y_fitted, c='r', label='Sigmoidal acumulated cases predictions')
	ax[1].plot(actual_cases, c='b', label='Actual Data')
	ax[1].axvline(x=cases_size, linestyle='--')
	ax[1].legend(loc='upper left')
elif option_1 == 'Deaths':
	#fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
	ax[0].set_title('Deaths')
	ax[0].plot(deaths_predict, c='r', label='LSTM acumulated deaths predictions')
	ax[0].plot(actual_deaths, c='b', label='Actual Data')
	ax[0].axvline(x=deaths_size, linestyle='--')
	ax[0].legend(loc='upper left')
	#
	ax[1].set_title('Sigmoidal')
	ax[1].plot(y_fitted, c='r', label='Sigmoidal acumulated cases predictions')
	ax[1].plot(actual_cases, c='b', label='Actual Data')
	ax[1].axvline(x=deaths_size, linestyle='--')
	ax[1].legend(loc='upper left')

######################################################

st.pyplot(fig)

# ESTO FUERA BUENO PARA EL TIEMPO

#streamlit.slider(label, min_value=None, max_value=None, value=None, step=None, format=None, key=None)	




