from functools import partial

import pandas as pd

import SignalResearch as sig

"""
Data Pre-processing
"""

# Assuming df is your DataFrame
df = pd.read_csv('data/data.csv')

# Make sure date column is datetime
df['date'] = pd.to_datetime(df['date'])

# Sort values to assure the date is in ascending order
df = df.sort_values(['ticker', 'date'])

# Calculate daily return as difference from previous day
df['daily_return'] = df.groupby('ticker')['last'].diff()

# To handle the first record of each stock which will be NaN after diff()
df['daily_return'] = df['daily_return'].fillna(0)

# add signals
pmt_short_term = partial(sig.price_momentum_signal, window=10)
pmt_mid_term = partial(sig.price_momentum_signal, window=30)
pmt_long_term = partial(sig.price_momentum_signal, window=90)

rsi_short_term = partial(sig.rsi_signal, window=10)
rsi_mid_term = partial(sig.rsi_signal, window=30)
rsi_long_term = partial(sig.rsi_signal, window=90)

price_acc_short_term = partial(sig.price_acceleration_signal, window=10)
price_acc_mid_term = partial(sig.price_acceleration_signal, window=30)
price_acc_long_term = partial(sig.price_acceleration_signal, window=90)

func_list = [pmt_long_term, pmt_mid_term, pmt_short_term, rsi_short_term, rsi_long_term, rsi_mid_term,
             price_acc_short_term, price_acc_long_term, price_acc_mid_term]
signal_list = ['pmt_long_term', 'pmt_mid_term', 'pmt_short_term', 'rsi_short_term',
               'rsi_long_term', 'rsi_mid_term', 'price_acc_short_term',
               'price_acc_long_term', 'price_acc_mid_term']

for k in range(len(func_list)):
    partial_func = func_list[k]
    df[signal_list[k]] = df.groupby('ticker')['last'].apply(partial_func)

func_list = [sig.vwap_signal, sig.pvt_signal, sig.cross_sectional_mean_reversion_signal]
for k in range(len(func_list)):
    partial_func = func_list[k]
    df[partial_func.__name__] = df.groupby('ticker').apply(partial_func).reset_index(level=0, drop=True)

# get the clip we buy
df['clip'] = 1

all_signals = ['pmt_long_term',
               'pmt_mid_term', 'pmt_short_term', 'rsi_short_term', 'rsi_long_term',
               'rsi_mid_term', 'price_acc_short_term', 'price_acc_long_term',
               'price_acc_mid_term', 'vwap_signal', 'pvt_signal',
               'cross_sectional_mean_reversion_signal']

# standardize the signals
for signal in all_signals:
    df[signal] = df.groupby('ticker')[signal].transform(lambda x: (x - x.mean()) / x.std())

# prevent forward looking
for signal in all_signals:
    df[signal] = df[signal].shift(1)
df.to_csv('data/crafted_data.csv')
