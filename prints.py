from libraries import *

def print_best_hyperparams(params: dict) -> None:
    """
    Prints the best hyperparameters from the optimization process.
    Parameters
    ----------
    params : dict
        Dictionary containing hyperparameter names and their best values.
    """
    print('\n--- Best Hyperparameters ---')
    for param, value in params.items():
        if isinstance(value, float):
            print(f'  {param}: {value:.4f}')
        else:
            print(f'  {param}: {value}')
    print("-----------------------------------\n")
