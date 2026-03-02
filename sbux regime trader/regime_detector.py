import pandas as pd
import numpy as np
import pandas_ta as ta   
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_clean_data() -> pd.DataFrame:
    path = Path("data/sbux_1h_clean.parquet")
    df = pd.read_parquet(path)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)  # drop 'Price'
    
    df = df.rename(columns=str.capitalize)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
    
    print(f"Loaded clean SBUX 1H data: {df.shape[0]:,} bars")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    return df

def detect_regime(df: pd.DataFrame, adx_period: int = 14, adx_threshold: float = 25.0) -> pd.DataFrame:
    df = df.copy()
    
    # Battle-tested ADX + ATR
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
    df['ADX'] = adx_df[f'ADX_{adx_period}']
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=adx_period)
    
    df['regime'] = np.where(df['ADX'] > adx_threshold, 'trending', 'range_bound')
    
    # Stats on valid bars only 
    valid = df['ADX'].notna()
    regime_pct = df.loc[valid, 'regime'].value_counts(normalize=True) * 100
    print("\nSBUX 1H Regime Breakdown (valid bars only):")
    print(f"Trending (→ Trend Following):   {regime_pct.get('trending', 0):.1f}%")
    print(f"Range-bound (→ Mean Reversion): {regime_pct.get('range_bound', 0):.1f}%")
    
    return df

#def save_regime_plot(df: pd.DataFrame):
    Path("results").mkdir(exist_ok=True)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name="SBUX 1H"), secondary_y=False)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name="ADX (14)", line=dict(color="purple")),
                  secondary_y=True)
    
    for regime, color in [('trending', 'rgba(0,200,0,0.12)'), ('range_bound', 'rgba(255,140,0,0.12)')]:
        mask = df['regime'] == regime
        if mask.any():
            fig.add_vrect(x0=df.index[mask].min(), x1=df.index[mask].max(),
                          fillcolor=color, opacity=0.4, layer="below", line_width=0)
    
    fig.update_layout(title="SBUX Regime Detection – ADX > 25 = Trending (Trend Following Mode)",
                      xaxis_title="Date (NY time)", yaxis_title="Price",
                      yaxis2_title="ADX", height=650, template="plotly_white")
    fig.write_html("results/sbux_regime_chart.html")
    print("Static regime chart saved → results/sbux_regime_chart.html (open in browser)")

if __name__ == "__main__":
    df = load_clean_data()
    df_regime = detect_regime(df)
    #save_regime_plot(df_regime)
    
    df_regime.to_parquet("data/sbux_1h_with_regime.parquet", compression="gzip")
    print("Regime data saved → data/sbux_1h_with_regime.parquet")