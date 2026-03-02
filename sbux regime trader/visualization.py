import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")

# ==================== CONFIG ====================
Path("results/plots").mkdir(parents=True, exist_ok=True)
FILE_PATH = "data/sbux_1h_backtest_results.parquet" 

# Load and fix any hidden MultiIndex
df = pd.read_parquet(FILE_PATH)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

print(f"Loaded {len(df):,} bars")
print("Columns:", df.columns.tolist())

# Bollinger & SMAs for visual
df['bb_mid'] = df['Close'].rolling(20).mean()
df['bb_upper'] = df['bb_mid'] + 2 * df['Close'].rolling(20).std()
df['bb_lower'] = df['bb_mid'] - 2 * df['Close'].rolling(20).std()
df['sma20'] = df['Close'].rolling(20).mean()
df['sma50'] = df['Close'].rolling(50).mean()
df['sma200'] = df['Close'].rolling(200).mean()

# Buy & Hold benchmark
initial_capital = 100000
df['equity_bh'] = (initial_capital / df['Close'].iloc[0]) * df['Close']

# Extract completed trades for histogram
trade_returns = df['trade_return'][df['trade_return'] != 0].dropna()
trade_pct = trade_returns / (df['position'].shift(1) * df['entry_price'].shift(1)) * 100
trade_pct = trade_pct.dropna()

print(f"{len(trade_pct)} completed trades for histogram")


# ====================== 1. Price with Signals ======================
print("Generating 01_price_with_signals.png ...")
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

mpf.plot(df, type='candle', style='yahoo', ax=ax, volume=False, 
         show_nontrading=False, warn_too_much_data=10000)

#ax.plot(df.index, df['bb_upper'], color='gray', lw=1, alpha=0.7, label='BB Upper')
#ax.plot(df.index, df['bb_lower'], color='gray', lw=1, alpha=0.7, label='BB Lower')
ax.plot(df.index, df['sma20'], color='#ff7f0e', lw=1, label='SMA 50') #orange
ax.plot(df.index, df['sma50'], color='#1f77b4', lw=1.5, label='SMA 50')         
ax.plot(df.index, df['sma200'], color='#2ca02c', lw=1.5, label='SMA 200')       #green
ax.plot(df.index, df['Close'], color='black', lw=1.5, label='Close' ),          #blue

# Bollinger Bands area fill + faint edges
ax.fill_between(
    df.index,
    df['bb_upper'],
    df['bb_lower'],
    color='#ff7f0e',
    alpha=0.10,
    zorder=1,
    label='Bollinger Band (±2σ)'
)

ax.plot(df.index, df['bb_upper'], color='#ff7f0e', lw=0.9, alpha=0.6, zorder=2)
ax.plot(df.index, df['bb_lower'], color='#ff7f0e', lw=0.9, alpha=0.6, zorder=2)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

plt.text(0.98, 0.02, f"Generated {datetime.now().strftime('%Y-%m')}", 
         transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='gray')

# Regime shading
for i in range(1, len(df)):
    if df['regime'].iloc[i] == 'trending' and df['regime'].iloc[i-1] != 'trending':
        start = df.index[i]
    elif df['regime'].iloc[i] != 'trending' and df['regime'].iloc[i-1] == 'trending':
        ax.axvspan(start, df.index[i], color='green', alpha=0.12)

# Signals
buy_idx = df[(df['mr_signal'] == 1) | (df['tf_signal'] == 1)].index
ax.scatter(buy_idx, df.loc[buy_idx, 'Close'], marker='^', color='#2ca02c', s=150, label='BUY', zorder=5)

exit_idx = df[(df['mr_exit'] == 1) | (df['tf_exit'] == 1)].index
ax.scatter(exit_idx, df.loc[exit_idx, 'Close'], marker='v', color='#d62728', s=150, label='EXIT', zorder=5)

ax.set_title(f"SBUX Hourly — Mean-Reversion + Trend-Following Signals (100 days)", fontsize=18)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Price ($)", fontsize=14)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig("results/plots/01_price_with_signals.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# ====================== 2. Equity Curve ======================
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.plot(df.index, df['equity'], color='#1f77b4', lw=3, label='Strategy')
ax.plot(df.index, df['equity_bh'], color='#ff7f0e', ls='--', lw=2, label='Buy & Hold')
ax.set_title("Equity Curve — Hybrid Strategy vs Buy & Hold", fontsize=18)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Equity ($)", fontsize=14)
ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig("results/plots/02_equity_curve.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# ====================== 3. Equity + Drawdown ======================
fig = plt.figure(figsize=(16, 9), dpi=300)
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

ax1.plot(df.index, df['equity'], color='#1f77b4', lw=3)
ax1.set_ylabel("Equity ($)")
dd = (df['equity'] / df['equity'].cummax() - 1) * 100
ax2.fill_between(df.index, dd, 0, color='#d62728', alpha=0.7)
ax2.set_ylabel("Drawdown (%)")
ax2.axhline(0, color='black', lw=0.5)

max_dd = dd.min()
ax2.annotate(f'Max DD: {max_dd:.1f}%', xy=(dd.idxmin(), max_dd), 
             xytext=(dd.idxmin(), max_dd-8), arrowprops=dict(arrowstyle='->', color='red'), fontsize=14)

fig.suptitle("Equity Curve & Underwater Drawdown", fontsize=18)
plt.tight_layout()
plt.savefig("results/plots/03_equity_drawdown.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
