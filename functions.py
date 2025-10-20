from libraries import *
from dataclasses import dataclass


@dataclass
class Position:
    """
    Represents a position in the portfolio.

    Attributes:
    n_shares : float
        Number of shares or units in the position.
    price : float
        Entry price of the position.
    sl : float
        Stop loss price.
    tp : float
        Take profit price.
    profit : float, optional
        Realized profit or loss of the position (default: None).
    exit_price : float, optional
        Exit price of the position (default: None).
    """
    n_shares: float
    price: float
    sl: float
    tp: float
    profit: float = None
    exit_price: float = None


@dataclass
class MLP_Params:
    """
    Parameters for MLP Classifier.
     Attributes:
     hidden_layer_sizes : tuple
        Size of hidden layers.
    activation : str
        Activation function.
    solver : str
        Solver for weight optimization.
    max_iter : int
        Maximum number of iterations.
    """

    hidden_layer_sizes: tuple = (64, 32)
    activation: str = 'relu'
    solver: str = 'adam'
    max_iter: int = 10000

@dataclass
class CNN_Params:
    """
    Parameters for CNN model.
    Attributes:
        lookback : int
            Number of previous time steps to consider.
        conv_layers : int
            Number of convolutional layers.
        filters : int
            Number of filters in the convolutional layers.
        kernel_size : int       
            Size of the convolutional kernels.
        dense_units : int
            Number of units in the dense layer.
        activation : str
            Activation function.
        optimizer : str
            Optimizer for training.
        epochs : int
            Number of training epochs.
        batch_size : int
            Size of training batches.
    """
    lookback: int = 20
    conv_layers: int = 2
    filters: int = 32
    kernel_size: int = 3
    dense_units: int = 64
    activation: str = 'relu'
    optimizer: str = 'adam'
    epochs: int = 10
    batch_size: int = 32


@dataclass
class BacktestingCapCOM:
    """
    Backtesting configuration with initial capital and commission.

    Attributes:
    initial_capital : float
        Initial capital for backtesting (default: 1_000_000).
    COM : float
        Commission per trade in percentage (default: 0.125 / 100).
    """
    initial_capital: float = 1_000_000
    COM: float = 0.125 / 100
    borrow_Rate: float = 0.25


@dataclass
class OptunaOpt:
    """
    Optuna optimization configuration.

    Attributes:
    direction : str
        Optimization direction ('maximize' or 'minimize').
    n_trials : int
        Number of optimization trials.
    n_jobs : int
        Number of parallel jobs (-1 uses all cores).
    n_splits : int
        Number of cross-validation splits.
    show_progress_bar : bool
        Show Optuna progress bar.
    """
    direction: str = 'maximize'
    n_trials: int = 50
    n_jobs: int = -1
    n_splits: int = 5
    show_progress_bar: bool = True



def get_portfolio_value(cash: float, long_ops: list[Position], short_ops: list[Position], current_price: float, n_shares: float) -> float:
    """
    Calculates the total portfolio value at a given moment.

    Parameters:
    cash : float
        Cash available in the portfolio.
    long_ops : list[Position]
        List of active long positions.
    short_ops : list[Position]
        List of active short positions.
    current_price : float
        Current price of the asset.
    n_shares : float
        Number of shares per position.

    Returns:
    float
        Total portfolio value including long and short positions.
    """
    value = cash
    for pos in long_ops:
        value += current_price * pos.n_shares
    for pos in short_ops:
        value += (pos.price * pos.n_shares) + \
            (pos.price - current_price) * pos.n_shares
    return value

