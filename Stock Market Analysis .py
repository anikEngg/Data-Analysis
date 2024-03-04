#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings, gc
import numpy as np 
import pandas as pd
import matplotlib.colors
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error,mean_absolute_error
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa import SARIMAX
#import statsmodels.tsa
from statsmodels.tsa.arima.model import ARIMA#, SARIMAX
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from sklearn.inspection import permutation_importance
import math
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

import xgboost as xgb


# In[ ]:


pip install --upgrade scikit-learn


# In[ ]:


pip install --upgrade pmdarima


# In[ ]:


pip install plotly


# In[2]:


pip install xgboost


# In[3]:


TD_PATH  = "D:\ML project\Stock Market Analysis\Dataset\TD.csv"
RY_PATH  = "D:\ML project\Stock Market Analysis\Dataset\RY.csv"
BNS_PATH = "D:\ML project\Stock Market Analysis\Dataset\BNS.csv"


# In[4]:


import pandas as pd

# Assuming TD_PATH, RY_PATH, and BNS_PATH are paths to your CSV files
# Read CSV files into DataFrames
td_df = pd.read_csv(TD_PATH, parse_dates=['date'], index_col='date')
ry_df = pd.read_csv(RY_PATH, parse_dates=['date'], index_col='date')
bns_df = pd.read_csv(BNS_PATH, parse_dates=['date'], index_col='date')

# Print the number of unique values in each column of td_df
print(td_df.nunique())

# Print the entire td_df DataFrame
print(td_df)


# In[5]:


td_date = td_df.index.unique()
ry_date = ry_df.index.unique()
bns_date = bns_df.index.unique()
td_date.shape, ry_date.shape, bns_date.shape


# ### Data Visualization

# In[6]:


####Closing price comparison
plt.figure(figsize=(10,6))
plt.grid(True, linewidth=0.5)

plt.plot(td_date, td_df['close'], label='TD', c='#00cc26', alpha=1, linewidth=1.5)
plt.plot(ry_date, ry_df['close'], label='RY', c='#ff336b', alpha=0.85, linewidth=1.5)
plt.plot(bns_date, bns_df['close'], label='BNS', c='#ffcc00', alpha=0.7, linewidth=1.5)

plt.xlabel('Year', weight="bold", size=12)
plt.ylabel('Close Price (CAD)', weight="bold", size=12)
plt.title('Closing Price Comparison', weight="bold", size=14)
plt.legend(fontsize="large", title="Canadian Banks", title_fontsize=12)
plt.show()


# In[7]:


###Closing price moving average


df = td_df.copy()
col = 'AdjustedClose'
periods = [5,10,20,30,50]
for period in periods:
    df.loc[:,"MovingAvg_{}Day".format(period)] = df['close'].rolling(window=period).mean().values



# In[8]:


plt.figure(figsize=(10,6))
plt.grid(True, linewidth=0.5)

plt.plot(td_date[6220:6520], df['close'][6220:6520], label='Close Price', alpha=1, linewidth=1.5)
plt.plot(td_date[6220:6520], df['MovingAvg_10Day'][6220:6520], label='10-day Average', linestyle='--', alpha=0.85, linewidth=1.5)
plt.plot(td_date[6220:6520], df['MovingAvg_30Day'][6220:6520], label='30-day Average', linestyle='--', alpha=0.85, linewidth=1.5)

plt.xlabel('Date', weight="bold", size=12)
plt.ylabel('Price (CAD)', weight="bold", size=12)
plt.title('Close Moving Average', weight="bold", size=14)
plt.legend(fontsize="large", title="Period", title_fontsize=12)
plt.show()


# In[16]:


pip install seaborn


# In[9]:


###Combining stock price + close moving average
plt.style.use('ggplot')
fig, axs = plt.subplots(1,2,figsize=(20,6), constrained_layout=True)

fig.suptitle('Stock Price Analysis', weight="bold", size=24)
axs[0].grid(True, linewidth=0.5)

axs[0].plot(td_date[6020:6520], td_df['close'][6020:6520], label='TD', c='#5FC144', alpha=1, linewidth=1.5)
axs[0].plot(ry_date[6240:6740], ry_df['close'][6240:6740], label='RY', c='#1741C6', alpha=0.85, linewidth=1.5)
axs[0].plot(bns_date[5250:5750], bns_df['close'][5250:5750], label='BNS', c='#FC0A3F', alpha=0.7, linewidth=1.5)

axs[0].set_xlabel('Date', weight="bold", size=16)
axs[0].set_ylabel('Price (CAD)', weight="bold", size=16)
axs[0].set_title('Closing Price Comparison', weight="bold", size=18)
axs[0].legend(fontsize="large", title="Canadian Banks", title_fontsize=12)
for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
  label.set_fontsize(14)

axs[1].grid(True, linewidth=0.5)

axs[1].plot(td_date[6220:6520], df['close'][6220:6520], label='Close Price', color='#0D22D6', alpha=1, linewidth=1.5)
axs[1].plot(td_date[6220:6520], df['MovingAvg_10Day'][6220:6520], label='10-day Average', linestyle='--', color='orange', alpha=1.0, linewidth=1.5)
axs[1].plot(td_date[6220:6520], df['MovingAvg_30Day'][6220:6520], label='30-day Average', linestyle='--', color='#C50AFC', alpha=0.95, linewidth=1.5)

axs[1].set_xlabel('Date', weight="bold", size=16)
axs[1].set_ylabel('Price (CAD)', weight="bold", size=16)
axs[1].set_title('TD Close Moving Average', weight="bold", size=18)
axs[1].legend(fontsize="large", title="Period", title_fontsize=12, loc='upper left')
axs[1].axvspan(*mdates.datestr2num(['8/5/2021','8/30/2021']), color='#0DD622', alpha=0.15)
axs[1].axvspan(*mdates.datestr2num(['3/20/2022','4/15/2022']), color='#0DD622', alpha=0.15)
axs[1].axvspan(*mdates.datestr2num(['5/25/2022','6/20/2022']), color='#0DD622', alpha=0.15)

for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
  label.set_fontsize(14)

plt.savefig('stock_analysis.png')
plt.show()


# ### Predictive Analysis-SARIMA

# In[10]:


####Determine if the data is seasonal or not
y = td_df.reset_index()[['date','close']][6045:6545]


# In[11]:


decomposition = sm.tsa.seasonal_decompose(y['close'], model='additive', period=10)
fig = decomposition.plot()
plt.show()


# ### SARIMA modeling

# In[12]:


##SARIMA modeling
# Start 31-08-2020 End 31-08-2022
td_df_subset_s = td_df[6045:6545].copy()
td_df_subset_s 


# In[13]:


train_data_s, test_data_s = td_df_subset_s[0:int(len(td_df_subset_s)*0.8)], td_df_subset_s[int(len(td_df_subset_s)*0.8):]


# In[14]:


smodel = auto_arima(td_df_subset_s['close'], start_p=1, start_q=1,
                    test='adf',
                    max_p=4, max_q=4,
                    m=30, #30 days, frequency
                    start_P=0, seasonal=True,
                    d=None, D=1, trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True)


# In[15]:


###Forecasting
n_periods = 30*3 # for the next 3 months
s_fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True) # forecast on a daily basic
index_of_fc = pd.date_range(td_df_subset_s.index[-1], periods = n_periods, freq='D')

# make series for plotting purpose
fitted_series = pd.Series(s_fitted.values, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)




# In[16]:


plt.style.use('seaborn-whitegrid')
fig1 = plt.figure(figsize=(20,10), constrained_layout=True)

axs = fig1.subplot_mosaic(
    """
    AB
    CC
    """
)

fig1.suptitle('TD Stock Price Analysis', weight="bold", size=24)
axs['A'].grid(True, linewidth=0.5)

axs['A'].plot(td_date[6020:6520], td_df['close'][6020:6520], label='TD', c='#5FC144', alpha=1, linewidth=1.5)
axs['A'].plot(ry_date[6240:6740], ry_df['close'][6240:6740], label='RY', c='#1741C6', alpha=0.85, linewidth=1.5)
axs['A'].plot(bns_date[5250:5750], bns_df['close'][5250:5750], label='BNS', c='#FC0A3F', alpha=0.7, linewidth=1.5)

axs['A'].set_xlabel('Date', weight="bold", size=18)
axs['A'].set_ylabel('Price (CAD)', weight="bold", size=18)
axs['A'].set_title('Closing Price Comparison', weight="bold", size=20)
axs['A'].legend(fontsize="large", title="Canadian Banks", title_fontsize=12)
for label in (axs['A'].get_xticklabels() + axs['A'].get_yticklabels()):
    label.set_fontsize(16)

axs['B'].grid(True, linewidth=0.5)

axs['B'].plot(td_date[6220:6520], df['close'][6220:6520], label='Close Price', color='#0D22D6', alpha=1, linewidth=1.5)
axs['B'].plot(td_date[6220:6520], df['MovingAvg_10Day'][6220:6520], label='10-day Average', linestyle='--', color='orange', alpha=1.0, linewidth=1.5)
axs['B'].plot(td_date[6220:6520], df['MovingAvg_30Day'][6220:6520], label='30-day Average', linestyle='--', color='#C50AFC', alpha=0.95, linewidth=1.5)

axs['B'].set_xlabel('Date', weight="bold", size=18)
axs['B'].set_ylabel('Price (CAD)', weight="bold", size=18)
axs['B'].set_title('TD Close Moving Average', weight="bold", size=20)
axs['B'].legend(fontsize="large", title="Period", title_fontsize=12, loc='upper left')
axs['B'].axvspan(*mdates.datestr2num(['8/5/2021','8/30/2021']), color='#0DD622', alpha=0.15)
axs['B'].axvspan(*mdates.datestr2num(['3/20/2022','4/15/2022']), color='#0DD622', alpha=0.15)
axs['B'].axvspan(*mdates.datestr2num(['5/25/2022','6/20/2022']), color='#0DD622', alpha=0.15)

for label in (axs['B'].get_xticklabels() + axs['B'].get_yticklabels()):
    label.set_fontsize(16)

axs['C'].plot(td_df_subset_s.index[125:], td_df_subset_s.close[125:], label='Observed', color='#0D22D6', alpha=1, linewidth=1.5)
axs['C'].plot(fitted_series, linestyle='--', label='Forecast', color='#F0083D', linewidth=1.5)
axs['C'].fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
for label in (axs['C'].get_xticklabels() + axs['C'].get_yticklabels()):
    label.set_fontsize(16)

axs['C'].set_xlabel('Date', weight="bold", size=18)
axs['C'].set_ylabel('Price (CAD)', weight="bold", size=18)
axs['C'].legend(loc='upper left',fontsize='large')
axs['C'].set_title("SARIMA - TD Closing Price Forecasting in 3 months", weight="bold", size=20 )


plt.savefig('stock_analysis.png')
plt.show()


# ### XGBoost

# In[ ]:


####Time Series Cross-Validation (CV)


# In[17]:


# Instantiate a Time Series cross-validator with test_size = 1 year with no gap
tts = TimeSeriesSplit(n_splits=5, test_size=365*1, gap=0)


# In[18]:


plt.style.use('seaborn-whitegrid')
fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True, constrained_layout=True)

fig.suptitle('Data Train/Test Split', weight="bold", size=20)

fold = 0
for train_idx, val_idx in tts.split(td_df[365*7:]):
    train = td_df[365*7:].iloc[train_idx]
    test = td_df[365*7:].iloc[val_idx]
    train['close'].plot(ax=axs[fold],
                        label='Training Set',
                        color='#0D22D6', linewidth=1.5,
                       )
    test['close'].plot(ax=axs[fold],
                       label='Test Set', c='orange')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    axs[fold].set_title(f'Fold {fold}', size=18, weight='bold')
    
    for label in (axs[fold].get_xticklabels() + axs[fold].get_yticklabels()):
        label.set_fontsize(12)
   
    fold += 1
axs[4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=True, shadow=True, ncol=5, prop={'size': 12, 'weight': "bold"})

axs[4].set_xlabel('Date', size=14, weight="bold")

plt.show()


# In[ ]:


###Forecasting Horizon


# In[ ]:


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(td_df[365*7:])


# In[ ]:


##Lag Features
def add_lags(df):
    target_map = df['close'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df


# In[ ]:


df = add_lags(df)


# In[ ]:


###Training
tss = TimeSeriesSplit(n_splits=5, test_size=365*1, gap=0)
df = df.sort_index()


fold = 0
preds = []
scores = []
reg = None
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2','lag3']
    TARGET = 'close'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)


# In[ ]:


print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')


# In[ ]:


###Feature importance

fig, ax = plt.subplots(1,1,figsize=(10,5))
xgb.plot_importance(reg, ax=ax, color='#4361ee')
ax.set_xlabel('F_score', weight="bold", size=14)
ax.set_ylabel('Features', weight="bold", size=14)
ax.set_title('Feature importance', weight='bold', size=14)
plt.rcParams.update({'font.size': 12})



# In[ ]:


###Predicting the future


# In[ ]:


##Feature Preparation
# Retrain on all data
df = create_features(df)

FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year',
            'lag1','lag2','lag3']
TARGET = 'close'

X_all = df[FEATURES]
y_all = df[TARGET]

reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators=500,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)


# In[ ]:


df.index.max()


# In[ ]:


# Create future dataframe
future = pd.date_range('2022-08-31','2023-08-29', freq='D')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)


# In[ ]:


df_and_future.tail()


# In[ ]:


future_w_features = df_and_future.query('isFuture').copy()


# In[ ]:


###Forecasting
future_w_features['pred'] = reg.predict(future_w_features[FEATURES])


# In[ ]:


train = df.copy()
test = df.copy()
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx].copy()
    test = df.iloc[val_idx].copy()


# In[ ]:


train.shape, test.shape


# In[ ]:


fig, axs = plt.subplots(1, 1, figsize=(20, 8), sharex=True, constrained_layout=True)
fig.suptitle('XGBoost-TD Closing Price Forecasting in 1 year', size=24, weight="bold")
train[2725:]['close'].plot(ax=axs, label='Training Set',
                            color='#0D22D6', linewidth=1.5)
test['close'].plot(ax=axs, label='Test Set', color='orange',
                   linewidth = 1.5)
future_w_features['pred'].plot(color='#F0083D', label='Forecast',
                               linewidth=1.5, linestyle='-'
                              )
axs.set_xlabel('Date', size=20, weight="bold")
axs.set_ylabel('Price (CAD)', size=20, weight="bold")
axs.legend(loc='upper left', prop={'size': 16})

for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(16)
plt.show()


# In[ ]:


###Model saving


# In[ ]:


# Save model
reg.save_model('model.json')


# In[ ]:


reg_new = xgb.XGBRegressor()
reg_new.load_model('model.json')

