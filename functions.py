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



def dateset_split(data: pd.DataFrame, train: float, test: float, validation: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training, testing, and validation sets.

    Parameters:
    data : pd.DataFrame
        Complete dataset to split.
    train : float
        Fraction of data to use for training.
    test : float
        Fraction of data to use for testing.
    validation : float
        Fraction of data to use for validation.

    Returns:
    tuple
        train_data, test_data, validation_data
    """
    n = len(data)
    train_size = int(n * train)
    test_size = train_size + int(n * test)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:test_size]
    validation_data = data.iloc[test_size:]

    return train_data, test_data, validation_data

