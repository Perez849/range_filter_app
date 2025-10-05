import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Range Filter Signals", layout="wide")
st.title("Range Filter [DW] - Señales de Compra")

# -----------------------------
# Lista de tickers
tickers = [
    "AAPL","MSFT","GOOGL","AMZN","FB",
    "TSLA","NVDA","JPM","V","DIS",
    "NFLX","PYPL","ADBE","INTC","CSCO",
    "CMCSA","PEP","KO","NKE","MRK",
    "WMT","PFE","ORCL","ABT","CRM"
]

# Input fechas
start_date = st.date_input("Fecha inicio", pd.to_datetime("2023-01-01"))
end_date = st.date_input("Fecha fin", pd.to_datetime("2025-09-30"))

# Descargar precios
st.info("Descargando datos de Yahoo Finance...")
data = yf.download(tickers, start=start_date, end=end_date)
close_prices = data['Close']
high_prices = data['High']
low_prices = data['Low']

# -----------------------------
# Parámetros del Range Filter
f_type = "Type 1"
rng_qty = 2.618
rng_scale = "Average Change"
rng_per = 14
smooth_range = True
smooth_per = 27

# -----------------------------
# Funciones
def cond_ema(x, cond, n):
    ema = np.zeros_like(x)
    val = []
    for i in range(len(x)):
        if cond[i]:
            val.append(x[i])
            if len(val) > 1:
                val.pop(0)
            ema[i] = (val[0] - ema[i-1])*(2/(n+1)) + ema[i-1] if i > 0 else val[0]
        else:
            ema[i] = ema[i-1] if i > 0 else x[i]
    return ema

def rng_size(x, scale, qty, n):
    ac = cond_ema(np.abs(x - np.roll(x, 1)), np.ones_like(x, dtype=bool), n)
    if scale == "Average Change":
        return qty * ac
    else:
        raise NotImplementedError("Solo Average Change implementado")

def rng_filter(high, low, rng_, n, type_, smooth, sn):
    rng_smooth = cond_ema(rng_, np.ones_like(rng_), sn)
    r = rng_smooth if smooth else rng_
    filt = np.zeros_like(r)
    filt[0] = (high[0] + low[0])/2
    fdir = np.zeros_like(r)
    
    for i in range(1, len(r)):
        filt[i] = filt[i-1]
        if type_=="Type 1":
            if high[i]-r[i] > filt[i-1]:
                filt[i] = high[i]-r[i]
            if low[i]+r[i] < filt[i-1]:
                filt[i] = low[i]+r[i]
        elif type_=="Type 2":
            if high[i] >= filt[i-1]+r[i]:
                filt[i] = filt[i-1] + np.floor(abs(high[i]-filt[i-1])/r[i])*r[i]
            if low[i] <= filt[i-1]-r[i]:
                filt[i] = filt[i-1] - np.floor(abs(low[i]-filt[i-1])/r[i])*r[i]
        
        if filt[i] > filt[i-1]:
            fdir[i] = 1
        elif filt[i] < filt[i-1]:
            fdir[i] = -1
        else:
            fdir[i] = fdir[i-1]
    
    hi_band = filt + r
    lo_band = filt - r
    return hi_band, lo_band, filt, fdir

# -----------------------------
results = {}
buy_signals = []

st.info("Calculando señales...")

for ticker in tickers:
    df_close = close_prices[ticker].dropna()
    if len(df_close) < 2:
        continue
    high = df_close.values
    low = df_close.values
    rng_ = rng_size(df_close.values, rng_scale, rng_qty, rng_per)
    
    hi_band, lo_band, filt, fdir = rng_filter(high, low, rng_, rng_per, f_type, smooth_range, smooth_per)
    
    df_result = pd.DataFrame({
        "Close": df_close,
        "Filter": filt,
        "HiBand": hi_band,
        "LoBand": lo_band,
        "Trend": fdir
    }, index=df_close.index)
    
    results[ticker] = df_result
    
    # Calcular duración de la última señal alcista
    trend_series = df_result['Trend'].values
    last_change_index = np.where(np.diff(np.sign(trend_series)) != 0)[0]
    last_signal_idx = last_change_index[-1]+1 if len(last_change_index) > 0 else 0
    
    if trend_series[-1] == 1:
        buy_signals.append((ticker, df_result.index[last_signal_idx]))

# Ordenar por señal más reciente
buy_signals = sorted(buy_signals, key=lambda x: x[1], reverse=True)

st.success("Señales calculadas!")

# -----------------------------
# Selector de tickers comprables
if buy_signals:
    ticker_names = [ticker for ticker, _ in buy_signals]
    selected_ticker = st.selectbox(
        "Selecciona un ticker para ver gráfico",
        ticker_names
    )

    # Mostrar fecha de última señal junto al selector
    last_signal_dates = {ticker: date for ticker, date in buy_signals}
    st.write(f"Última señal de compra: {last_signal_dates[selected_ticker].date()}")

    # Mostrar gráfico del ticker seleccionado
    df = results[selected_ticker]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['Close'], color='black', label='Close')
    
    trend = df['Trend'].values
    filt = df['Filter'].values
    for i in range(1, len(trend)):
        color = 'green' if trend[i] == 1 else 'red'
        ax.plot(df.index[i-1:i+1], filt[i-1:i+1], color=color, linewidth=2)
    
    ax.set_title(f"{selected_ticker} - Señal de compra según Range Filter [DW]")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Ningún ticker cumple actualmente la señal de compra.")
