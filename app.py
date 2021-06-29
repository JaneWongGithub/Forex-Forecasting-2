

import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
import yfinance as yf
import datetime
from yahoofinancials import YahooFinancials

st.title('📈 Automated FOREX USD-AUD Forecasting')


"""
###upload Live Data directly from Yahoo Financials
"""
import pandas_datareader as pdr
from datetime import datetime
current_date = datetime.today()
import matplotlib.pyplot as plt


#data obtained from Yahoo Financials
#define variable for start and end time
start = datetime(2007, 1, 1)
end = current_date
USDAUD_data = yf.download('AUD=X', start, end)
USDAUD_data.head()
df = pd.dataframe(USDAUD_data)
plt.figure(figsize=(10, 7))
plt.plot(USDAUD_data)       
plt.title('USDAUD Prices')


USDAUD1 = df.drop(column=['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
USDAUD1_data.head

USDAUD2 = USDAUD1_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
USDAUD2.head                  
                  

"""
### Step 2: Select Forecast Horizon

Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 7)



"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""

future = m.make_future_dataframe(periods=periods_input)
    
forecast = m.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fcst_filtered =  fcst[fcst['ds'] > max_date]    
st.write(fcst_filtered)
    
"""
The next visual shows the actual (black dots) and predicted (blue line) values over time.
"""
fig1 = m.plot(forecast)
st.write(fig1)

"""
The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
"""
fig2 = m.plot_components(forecast)
st.write(fig2)

#using AR-Net
model = NeuralProphet(
    n_forecasts=60,
    n_lags=60,
    changepoints_range=0.95,
    n_changepoints=100,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    batch_size=64,
    epochs=100,
    learning_rate=1.0,
)

model.fit(USDAUD2, 
          freq='D',
          valid_p=0.2,
          epochs=100)

plot_forecast(model, USDAUD2, periods=60, historic_pred=True)

plot_forecast(model, USDAUD2, periods=60, historic_pred=False, highlight_steps_ahead=60)


