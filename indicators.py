from libraries import *
from dataclasses import dataclass

def indicators(data: pd.DataFrame) -> pd.DataFrame:

    close = data["Close"].squeeze()

    # Momentum Indicatos
        # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        # Stochastic Oscillator
    data['Stochastic_Oscillator'] = ta.momentum.StochasticOscillator(
        data['High'], data['Low'], data['Close'], window=14, smooth_window=3).stoch()
        # MACD
    data['MACD'] = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9).macd()
        # CCI
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=20).cci()
        # Williams %R
    data['Williams_%R'] = ta.momentum.WilliamsRIndicator(
        data['High'], data['Low'], data['Close'], lbp=14).williams_r()
        # ROC
    data['ROC'] = ta.momentum.ROCIndicator(data['Close'], window=12).roc()
        # Awesome Oscillator
    data['Awesome_Oscillator'] = ta.momentum.AwesomeOscillator(
        data['High'], data['Low'], window1=5, window2=34).awesome_oscillator()
        # Classic Momentun Indicator
    data['CMI'] = ta.momentum.ClassicMomentumIndicator(data['Close'], window=14).cmi()

    # Volatility Indicators
        # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['bollinger_upper'] = bollinger.bollinger_hband()
    data['bollinger_lower'] = bollinger.bollinger_lband()
        # Average True Range (ATR)
    data['ATR'] = ta.volatility.AverageTrueRange(
        data['High'], data['Low'], data['Close'], window=14).average_true_range()
        # Keltner Channels
    keltner = ta.volatility.KeltnerChannel(
        data['High'], data['Low'], data['Close'], window=20, window_atr=10)
    data['keltner_upper'] = keltner.keltner_channel_hband()
    data['keltner_lower'] = keltner.keltner_channel_lband()
        # Donchian Channels
    donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], window=20)
    data['donchian_upper'] = donchian.donchian_channel_hband()
    data['donchian_lower'] = donchian.donchian_channel_lband()
        # Chaikin Volatility
    data['Chaikin_Volatility'] = ta.volatility.ChaikinVolatility(
        data['High'], data['Low'], window=10).chaikin_volatility()
    
    # Volume Indicators
        # On-Balance Volume (OBV)
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        # Chaikin Money Flow (CMF)
    data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
        data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
        # Volume Price Trend (VPT)
    data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
        # Money Flow Index (MFI)
    data['MFI'] = ta.volume.MFIIndicator(
        data['High'], data['Low'], data['Close'], data['Volume'], window=14).money_flow_index()
    
    data["Return"] = close.pct_change()
    data = data.dropna()

    return data


    