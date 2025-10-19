from libraries import *
from dataclasses import dataclass

def indicadores_prueba(data):

    close = data["Close"].squeeze()

    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()


    return data
