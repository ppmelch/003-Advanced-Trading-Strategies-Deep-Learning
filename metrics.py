import pandas as pd
import numpy as np

def sharpe_ratio(port_value: pd.Series) -> float:
    """
    Calcula el ratio de Sharpe anualizado de una serie temporal de valores de portafolio.
    Args:
        port_value (pd.Series): Serie temporal que contiene los valores del portafolio a lo largo del tiempo.
    Returns:
        float: El ratio de Sharpe anualizado.
    """
    returns = port_value.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    mu_ann = mu * (365 * 24)
    sigma_ann = sigma * np.sqrt(365 * 24)
    if sigma_ann > 0:
        sharpe = mu_ann / sigma_ann
    else:
        sharpe = 0
    return sharpe

def sortino_ratio(port_value: pd.Series) -> float:
    """
    Calcula el ratio de Sortino anualizado de una serie temporal de valores de portafolio.
    Args:
        port_value (pd.Series): Serie temporal que contiene los valores del portafolio a lo largo del tiempo.
    Returns:
        float: El ratio de Sortino anualizado.
    """
    returns = port_value.pct_change().dropna()
    mean_ret = returns.mean()
    downside = np.minimum(returns ,0).std()

    mean_ann = mean_ret * (365 * 24)
    downside_std_ann = downside * np.sqrt(365 * 24)
    if downside_std_ann > 0:
        sortino = mean_ann / downside_std_ann
    else:
        sortino = 0
    return sortino

def maximum_drawdown(port_value: pd.Series) -> float:
    """
    Calcula la máxima caída (drawdown) de una serie temporal de valores de portafolio.
    Args:
        port_value (pd.Series): Serie temporal que contiene los valores del portafolio a lo largo del tiempo.
    Returns:
        float: La máxima caída (drawdown) como un valor absoluto.
    """
    peaks = port_value.cummax()
    dd = (port_value - peaks) / peaks 
    maximum_dd = dd.min() 
    return abs(maximum_dd)

def calmar_ratio(port_value: pd.Series) -> float:
    """
    Calcula el ratio de Calmar anualizado de una serie temporal de valores de portafolio.
    Args:
        port_value (pd.Series): Serie temporal que contiene los valores del portafolio a lo largo del tiempo.
    Returns:
        float: El ratio de Calmar anualizado.
    """
    returns = port_value.pct_change().dropna()
    mean_ann = returns.mean() * (24 * 365)
    mdd = maximum_drawdown(port_value) 

    if mdd > 0:
        calmar = mean_ann / mdd
    else:
        calmar = 0.0
    return calmar

def win_rate(closed_positions) -> float:

    if not closed_positions:
        return 0.0
    n_wins = sum(
        1 for pos in closed_positions if pos.profit is not None and pos.profit > 0)
    return n_wins / len(closed_positions)



def metrics(port_value: pd.Series) -> pd.DataFrame:
    """
    Calcula y devuelve un DataFrame con varias métricas financieras clave
    basadas en una serie temporal de valores de portafolio.
    Args:
        port_value (pd.Series): Serie temporal que contiene los valores del portafolio a lo largo del tiempo.
    Returns:
        pd.DataFrame: DataFrame que contiene las métricas calculadas.
    """
    metrics_df = pd.DataFrame({
        'Sharpe Ratio': [sharpe_ratio(port_value)],
        'Sortino Ratio': [sortino_ratio(port_value)],
        'Maximum Drawdown': [maximum_drawdown(port_value)],
        'Calmar Ratio': [calmar_ratio(port_value)],
        'Win Rate': [win_rate(port_value)],
    }, index = ["Metrics"])
    return metrics_df

