import numpy as np

"""
The Signal Factory
Define functions that generates signals
"""


def price_momentum_signal(data, window=10):
    """
    calcualte momentum signal
    :param data: dataframe, must have column 'last'
    :param window: lookback, 10 by default
    :return: the constructed price momentum signal
    """
    return data.diff(window) / data.shift(window)


def rsi_signal(data, window):
    """
    calculate rsi signal
    :param data: a pandas dataframe, must contain 'last' column
    :param window: look back window, 14 by default
    :return: the constructed rsi signal
    """
    delta = data.diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    average_gain = up.rolling(window).mean()
    average_loss = abs(down.rolling(window).mean())

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)

    return rsi


def price_acceleration_signal(data, window=30):
    """
    calculate price acceleration signal
    :param data: dataframe, must have column 'last' or 'daily_return'
    :param window: look back, 30 by default
    :return: return the price acc signal
    """
    return data.diff().diff(window)


def vwap_signal(data):
    """
    calculate volume weighted average price (vwap)
    :param data: same
    :return: same
    """
    vwap = (data['volume'] * data['last']).cumsum() / data['volume'].cumsum()
    return vwap


def pvt_signal(data):
    """
    calculate price_volume_trend signal
    :param data: same
    :return: same
    """
    pvt = (data['last'].pct_change() + 1).cumprod() * data['volume']
    return pvt


def obv_signal(data):
    """
    calculate on balance volume (OBV) signal
    :param data: same
    :return: same
    """
    obv = np.where(data['last'] > data['last'].shift(), data['volume'],
                   np.where(data['last'] < data['last'].shift(), -data['volume'], 0)).cumsum()
    return obv


def cross_sectional_mean_reversion_signal(data):
    """
    calculate cross sectional mean reversion signal
    :param data: same
    :return: same
    """
    mean_return = data['last'].pct_change().mean()
    return data['last'].pct_change() - mean_return
