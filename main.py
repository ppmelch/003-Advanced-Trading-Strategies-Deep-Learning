from libraries import *
from functions import dateset_split 
from indicators import indicadores_prueba

data = yf.download("AZO", start="2010-10-10", end="2025-10-10", progress=False)["Close"]

train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

def main():
    pass



if __name__ == "__main__":
    main()