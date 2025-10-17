from libraries import *

data = yf.download("AZO", start="2010-10-10", end="2025-10-10")["Close"]

#train, test, validation = dataset_split(data, 0.6, 0.2, 0.2)

