import pandas as pd
from p_modules import m_modules as pp

USA = pd.read_csv('data/us_counties_covid19_daily.csv.zip')

cases = pp.Prep_cases(USA)
deaths = pp.Prep_deaths(USA)

cases.to_csv('data/acumulated_cases.csv')
deaths.to_csv('data/acumulated_deaths.csv')
