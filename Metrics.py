import matplotlib.pyplot as plt
import numpy as np

"""
Metrics used to assess the performance
"""


def metric(pnl):
    """
    evaluate various metrics with the provided pnl vector
    :param pnl: an iterator
    :return: a dictionary that maps each name of the metric to its evaluated value
    """
    # final pnl
    final_pnl = pnl[-1] - pnl[0]
    # sharpe ratio
    # calculate returns
    returns = np.diff(pnl) / pnl[:-1]

    # calculate mean return and standard deviation of return
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # assume 252 trading days in a year
    sharpe_ratio = np.sqrt(252) * mean_return / std_return
    # ddpl (max draw down over pnl)
    # Calculate the running maximum
    running_max = np.maximum.accumulate(pnl)
    # Calculate the drawdown as the difference between running max and current portfolio value
    drawdown = running_max - pnl
    # Calculate the maximum drawdown
    max_drawdown = np.max(drawdown)
    ddpl = max_drawdown / (pnl[-1] - pnl[0])

    metrics_tab = {}
    metrics_tab['pnl'] = final_pnl
    metrics_tab['sharpe_ratio'] = sharpe_ratio
    metrics_tab['ddpl'] = ddpl
    return metrics_tab


def plot_pnl(pnl_curves):
    pnl_curves.plot()

    plt.title('Comparison of PNL Curves')
    plt.xlabel('Time')
    plt.ylabel('PNL')

    plt.show()
