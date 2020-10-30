

#https://docs.streamlit.io/en/stable/api.html

#
#https://github.com/whiteboxml/teaching-ironhack-dataptmad-2004/blob/master/week_10/introduction_to_matplotlib_and_seaborn/introduction_to_matplotlib_and_seaborn.ipynb

#DEPENDENCIES

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
#%matplotlib inline


#MODULES
from p_modules import m_modules as pp
from p_analysis import m_analysis as an


#########################################################3

#INTRODUCTION

st.title('Introduction')
st.text('This is some text.')

st.markdown('Streamlit is **_really_ cool**.')

st.header('This is a header')
st.subheader('This is a subheader')

########################################################

# METODOLOGY

st.title('Metodology')
st.text('This is some text.')
st.write("Josue")

###########

# Por si acaso

#df = pd.DataFrame(
#    np.random.randn(50, 20),
#    columns=('col %d' % i for i in range(20)))

#st.dataframe(df)  # Same as st.write(df)

#############

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')
     
#image = ??????
#st.image(image, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', output_format='auto', **kwargs)

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')
     

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

st.graphviz_chart('''
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
''')


##########################################################

#DEMO
st.title('USA Coronavirus evolution')

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

# ANALISIS

#growth = an.growth_factor(cases_predict)
#an.growth_ratio(confirmed)


# PLOT

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
if  option_1 == 'Cases':
	#fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
	ax[0].set_title('Cases')
	ax[0].plot(cases_predict, c='r', label='LSTM acumulated cases predictions')
	ax[0].plot(actual_cases, c='b', label='Actual Data')
	ax[0].axvline(x=cases_size, linestyle='--')
	ax[0].legend(loc='upper left')
	#ax[1].set_title('Deaths')
	#ax[1].plot(deaths_predict, c='r', label='LSTM acumulated deaths predictions')
	#ax[1].plot(actual_deaths, c='b', label='Actual Data')
	#ax[1].axvline(x=deaths_size, linestyle='--')
	#ax[1].legend(loc='upper left')
elif option_1 == 'Deaths':
	#fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
	ax[0].set_title('Deaths')
	ax[0].plot(deaths_predict, c='r', label='LSTM acumulated deaths predictions')
	ax[0].plot(actual_deaths, c='b', label='Actual Data')
	ax[0].axvline(x=deaths_size, linestyle='--')
	ax[0].legend(loc='upper left')
	#ax[1].set_title('Deaths')
	#ax[1].plot(deaths_predict, c='r', label='LSTM acumulated deaths predictions')
	#ax[1].plot(actual_deaths, c='b', label='Actual Data')
	#ax[1].axvline(x=deaths_size, linestyle='--')
	#ax[1].legend(loc='upper left')


######################################################

#ax[0][2].set_title('Growth Factor')
#ax[0][2].plot(growth, c='r', label='Growth Factor')
#ax[0][2].plot(actual_deaths, c='b', label='Actual Data')
#ax[0][2].axvline(x=cases_size, linestyle='--')
#ax[0][2].legend(loc='upper left')

st.pyplot(fig)

# ESTO FUERA BUENO PARA EL TIEMPO

#streamlit.slider(label, min_value=None, max_value=None, value=None, step=None, format=None, key=None)	




