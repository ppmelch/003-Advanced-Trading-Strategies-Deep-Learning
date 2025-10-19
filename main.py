from libraries import *
from functions import dateset_split 
from indicators import indicadores_prueba
from sklearn.preprocessing import StandardScaler


data = yf.download("AZO", start="2010-10-10", end="2025-10-10", progress=False)["Close"]

target = data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

data = data.dropna()

X = data[["SMA_10", "SMA_50", "RSI", "Return"]]   # AQUI VAN LOS INDICADORES
y = target

X_train, X_test, X_validation = dateset_split(X, 0.6, 0.2, 0.2)
y_train, y_test, y_validation = dateset_split(y, 0.6, 0.2, 0.2)

def main():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_validation_scaled = scaler.transform(X_validation)

    

    pass



if __name__ == "__main__":
    main()