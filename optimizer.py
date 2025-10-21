from libraries import *

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
