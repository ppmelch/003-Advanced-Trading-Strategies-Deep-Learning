from libraries import *
from indicators import Indicators
from models import model_MLP, model_CNN
from prints import print_best_hyperparams
from sklearn.preprocessing import StandardScaler
from optimizer import optimize_hyperparams, dateset_split
from functions import CNN_Params, BacktestingCapCOM, OptunaOpt


data = yf.download("AZO", start="2010-10-10", end="2025-10-10", progress=False)

data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

data = Indicators.indicators(data)

X = data.drop(columns=["Target"])
y = data["Target"]

X_train, X_test, X_validation = dateset_split(X, 0.6, 0.2, 0.2)
y_train, y_test, y_validation = dateset_split(y, 0.6, 0.2, 0.2)

optimization_metric = "Calmar"  # Options: 'Sharpe', 'Sortino', 'Calmar'

def main():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_validation_scaled = scaler.transform(X_validation)

    print("Training MLP model...")
    results_mlp = model_MLP(X_train_scaled, X_test_scaled,
                            X_validation_scaled, y_train, y_test, y_validation)
    print(results_mlp)

    print("Training CNN model...")
    results_cnn = model_CNN(
        X_train_scaled, y_train, lookback=CNN_Params.lookback, params=CNN_Params.__dict__, name="CNN Trading")
    print(results_cnn)

    # --- CONFIG ---
    backtest_config = BacktestingCapCOM()
    optimizacion_config = OptunaOpt()

    # --- OPTUNA TRAIN MLP ---
    print("Starting Optuna optimization for MLP...")

    study_MLP = (optimize_hyperparams(
        X_train_scaled, y_train, backtest_config, optimizacion_config, optimization_metric
    ))
    best_params = study_MLP.best_trial

    print(
        f"--- Best {optimization_metric}: {study_MLP.best_trial.value:.4f} ---")
    print_best_hyperparams(best_params.params)

    # --- OPTUNA TRAIN CNN ---
    print("Starting Optuna optimization for CNN...")
    study_CNN = (optimize_hyperparams(
        X_train_scaled, y_train, backtest_config, optimizacion_config, optimization_metric
    ))
    best_params = study_CNN.best_trial

if __name__ == "__main__":
    main()
