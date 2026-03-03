# Regime-Switching Hybrid Trading System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/pandas-2.x-orange)
![Backtesting](https://img.shields.io/badge/Backtesting-Event--driven-green)

**Mean-Reversion + Trend-Following on Hourly SBUX Data**

An algorithmic trading project in Python that switches between mean-reversion and trend-following strategies based on ADX regime detection.

<img width="4770" height="2671" alt="02_equity_curve" src="https://github.com/user-attachments/assets/e7f0e1ee-2864-4555-8044-380e2633ba24" />
*Equity curve comparison: Strategy (blue) and buy & hold (orange) over the test period.*

---

## 📌 Project Overview

This project implements a **regime-aware hybrid trading framework** that dynamically allocates between:
- **Mean Reversion** — in range-bound (low ADX) markets
- **Trend Following** — in trending (high ADX) markets

The core motivation draws from a 2021 LinkedIn article exploring a simple Bollinger Bands + RSI mean-reversion strategy in Python. The author concluded:

> “This mean reversion strategy has the potential to deliver consistent returns in range-bound and volatile environments. However, it might not be the best strategy to implement in trending markets. It is also highly vulnerable to a stock market crash.”

They highlighted that pure mean-reversion performs well in sideways markets but can miss extended trends (generating few/no signals) and suffer large losses during sharp reversals or crashes. As a natural extension, they suggested blending mean reversion with trend-following to create a more adaptive system.

This project implements that idea using **ADX(14)** to detect regimes and route signals:

- Low ADX (≤ 25) → mean-reversion logic (buy oversold dips expecting reversion)  
- High ADX (> 25) → trend-following logic (enter on momentum with volume confirmation and ride with trailing ATR stop)  

The hybrid aims to capture opportunities across more market conditions while reducing exposure to regime-specific pitfalls of standalone mean-reversion.

---

## 🎯 Strategy Logic

1. **Regime Detection** (`regime_detector.py`)
   - ADX(14) > 25 → trending regime → trend-following logic
   - ADX(14) ≤ 25 → range-bound regime → mean-reversion logic
   - ATR(14) is calculated for use in position sizing and trailing stops
     
   - Code:
     ```python
     def detect_regime(df: pd.DataFrame, adx_period: int = 14, adx_threshold: float = 25.0) -> pd.DataFrame:
        df = df.copy()
    
        # Battle-tested ADX + ATR
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
        df['ADX'] = adx_df[f'ADX_{adx_period}']
    
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=adx_period)
    
        df['regime'] = np.where(df['ADX'] > adx_threshold, 'trending', 'range_bound')
     ```

2. **Mean Reversion** (`mean_reversion.py`) — active in range-bound regimes only
   - Entry: z-score < -2.0 (price at least 2 standard deviations below 20-period mean)
   - Exit: z-score > 0 or z-score > 3.0
     
   - Code:
     ```python
      def mean_reversion_logic(df: pd.DataFrame, 
                        z_entry: float = 2.0, 
                        z_stop: float = 3.0) -> pd.DataFrame:
          df = df.copy()
          df = calculate_zscore(df, window=20)

          df['mr_active'] = df['regime'] == 'range_bound'
    
          # Build signals with lists (no iloc loop bugs)
          n = len(df)
          mr_signal = [0] * n
          mr_exit = [0] * n
          mr_position = [0] * n
    
          in_position = False
          for i in range(n):
              curr_z = df.iloc[i]['zscore']
              curr_active = df.iloc[i]['mr_active']
        
              if not in_position:
                  # Entry: deep negative z-score + range-bound only
                  if curr_z < -z_entry and curr_active and not pd.isna(curr_z):
                      mr_signal[i] = 1
                      in_position = True
                      mr_position[i] = 1
              else:
                  # Exit while in trade
                  if curr_z > z_stop or curr_z > 0 or pd.isna(curr_z):
                      mr_exit[i] = 1
                      in_position = False
                      mr_position[i] = 0
                  else:
                      mr_position[i] = 1
    
          df['mr_signal'] = mr_signal
          df['mr_exit'] = mr_exit
          df['mr_position'] = mr_position
          
          entries = sum(mr_signal)
          exits = sum(mr_exit)

          print(f"\nMean Reversion Strategy (range-bound only, SBUX 1H):")
          print(f"   Long entries: {entries}")
          print(f"   Exits (z-stop or mean cross): {exits}")
          print(f"   Open trades at end: {mr_position[-1]}")
                  
          return df
     ```

3. **Trend Following** (`trend_following.py`) — active in trending regimes only
   - Entry: SMA(10) > SMA(50) and volume > 1.5 × 20-period average
   - Exit: price falls below trailing stop (highest high since entry − 3.0 × ATR)
     
   - Code:
     ```python
      def trend_following_logic(df: pd.DataFrame, 
                               fast: int = 10, 
                               slow: int = 50, 
                               atr_mult: float = 3.0) -> pd.DataFrame:
          df = df.copy()
    
          # MA crossover
          df['ma_fast'] = df['Close'].rolling(fast).mean()
          df['ma_slow'] = df['Close'].rolling(slow).mean()
          
          # Volume confirmation (spike above 20-bar average)
          df['vol_avg'] = df['Volume'].rolling(20).mean()
          df['vol_spike'] = df['Volume'] > df['vol_avg'] * 1.5
          
          # Regime gate
          df['tf_active'] = df['regime'] == 'trending'
          
          # Signals & state
          n = len(df)
          tf_signal = [0] * n
          tf_exit = [0] * n
          tf_position = [0] * n
          trailing_stop = [0.0] * n
          
          in_position = False
          highest_since_entry = 0.0
          
          for i in range(n):
              curr_close = df.iloc[i]['Close']
              curr_active = df.iloc[i]['tf_active']
              curr_vol_spike = df.iloc[i]['vol_spike']
              curr_above_ma = df.iloc[i]['ma_fast'] > df.iloc[i]['ma_slow']
        
              if not in_position:
                  if curr_above_ma and curr_active and curr_vol_spike and not pd.isna(curr_close):
                      tf_signal[i] = 1
                      in_position = True
                      highest_since_entry = curr_close
                      tf_position[i] = 1
                      trailing_stop[i] = curr_close - df.iloc[i]['ATR'] * atr_mult
              else:
                  # Update trailing stop
                  highest_since_entry = max(highest_since_entry, curr_close)
                  new_stop = highest_since_entry - df.iloc[i]['ATR'] * atr_mult
                  trailing_stop[i] = new_stop
                  
                  # Exit if price hits trailing stop
                  if curr_close < new_stop:
                      tf_exit[i] = 1
                      in_position = False
                      tf_position[i] = 0
                  else:
                      tf_position[i] = 1
    
          df['tf_signal'] = tf_signal
          df['tf_exit'] = tf_exit
          df['tf_position'] = tf_position
          df['trailing_stop'] = trailing_stop
          
          entries = sum(tf_signal)
          exits = sum(tf_exit)
          
          print(f"\nTrend Following Strategy (trending regimes only, SBUX 1H):")
          print(f"   Long entries: {entries}")
          print(f"   Exits (ATR trailing stop): {exits}")
          print(f"   Open trades at end: {tf_position[-1]}")
          
          return df
     ```
4. **Risk Rules** (`risk_manager.py`)
   - Risk per trade: 0.5%
   - Position size determined by ATR (stop distance = 1.5 × ATR)
   - Portfolio drawdown limit: 15% (trading pauses if exceeded)
   - Long positions only (no short exposure)
     
   - Code:
     ```python
      def calculate_position_size(self, atr: float, price: float) -> int:
          if pd.isna(atr) or atr <= 0 or self.paused:
              return 0
        
          risk_amount = self.current_equity * self.risk_per_trade
          stop_distance = atr * 1.5  # conservative buffer
          shares = int(risk_amount / stop_distance)
          return max(1, shares)  # at least 1 share if risk allows
     ```

5. **Backtesting Approach** (`backtester.py`)
   - Event-driven simulation with position carry-forward
   - Sharpe ratio calculated on daily returns
   - Chronological 70/30 in-sample / out-of-sample split
     
   - Code:
     ```python
      def run_backtest(df: pd.DataFrame, initial_capital: float = 100_000) -> pd.DataFrame:
          """Combined regime-switching backtest with RiskManager + slippage."""
          df = df.copy()
          rm = RiskManager(initial_capital=initial_capital)
          
          df['position'] = 0
          df['entry_price'] = 0.0
          df['equity'] = float(initial_capital)
          df['trade_return'] = 0.0
          
          n = len(df)
          position = 0
          entry_price = 0.0
          
          for i in range(n):
              row = df.iloc[i]
              price = row['Close']
              atr = row['ATR']
              
              # Carry forward equity & position every bar
              df.iloc[i, df.columns.get_loc('equity')] = rm.current_equity
              df.iloc[i, df.columns.get_loc('position')] = position
              
              # Skip if no valid ATR (warm-up period)
              if pd.isna(atr) or atr <= 0:
                  continue
              
              # RiskManager check BEFORE any entry
              if rm.can_trade():
                  size = rm.calculate_position_size(atr, price)
                  
                  # Only attempt entry if size > 0 (risk manager allows it)
                  if size > 0:
                      # Regime switch — 100% one strategy at a time
                      if row['regime'] == 'range_bound' and row['mr_signal'] == 1 and position == 0:
                          df.iloc[i, df.columns.get_loc('position')] = size
                          position = size
                          entry_price = price
                          df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                          rm.update_equity(rm.current_equity)
                          
                      elif row['regime'] == 'trending' and row['tf_signal'] == 1 and position == 0:
                          df.iloc[i, df.columns.get_loc('position')] = size
                          position = size
                          entry_price = price
                          df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                          rm.update_equity(rm.current_equity)
              
              # Exit logic (MR or TF) — checked every bar if in position
              if position > 0:
                  exit_condition = False
                  if row['regime'] == 'range_bound' and row['mr_exit'] == 1:
                      exit_condition = True
                  elif row['regime'] == 'trending' and row['tf_exit'] == 1:
                      exit_condition = True
                  elif row['regime'] == 'trending' and price < row['trailing_stop']:
                      exit_condition = True
                  
                  if exit_condition:
                      # Simple slippage simulation
                      exit_price = price * (1 - 0.0005)
                      trade_pnl = position * (exit_price - entry_price)
                      df.iloc[i, df.columns.get_loc('trade_return')] = trade_pnl
                      rm.current_equity += trade_pnl
                      rm.update_equity(rm.current_equity)
                      position = 0
                      entry_price = 0.0
              
              # Always carry forward position & equity
              df.iloc[i, df.columns.get_loc('position')] = position
              df.iloc[i, df.columns.get_loc('equity')] = rm.current_equity
          
          return df
     ```
---

## 📊 Performance Results (SBUX 1H, Oct 2025 – Feb 2026)

| Metric                  | Full Period   | In-Sample (70%) | Out-of-Sample (30%) |
|-------------------------|---------------|-----------------|---------------------|
| Total Return            | +5.8%         | +4.8%           | +2.8%               |
| Daily Sharpe Ratio      | 1.45          | 1.60            | 1.97                |
| Max Drawdown            | -0.8%         | -0.8%           | -0.3%               |
| Win Rate                | 73%           | 71.4%           | 66.7%               |
| Profit Factor           | 5.1           | 5.46            | 10.60               |
| Number of Trades        | 18            | 7               | 3                   |

The equity curve shows lower volatility than buy-and-hold but lower cumulative return over the period.

<img width="4770" height="2670" alt="03_equity_drawdown2" src="https://github.com/user-attachments/assets/1852574f-25a0-4457-9f0c-ddc7626d89ab" />
*Equity progression (top) and underwater drawdown (bottom). Maximum drawdown reached -0.8%.*

<img width="4761" height="2670" alt="01_price_with_signals" src="https://github.com/user-attachments/assets/d2db48e1-1a91-41f2-b06f-111289a01c84" />
*Price series with entry (green ▲) and exit (red ▼) markers. Background indicates regime classification periods.*

---

## ⚠️ Observed Limitations

Testing shows a clear regime-dependent performance asymmetry:

> The strategy achieves a Sharpe ratio around 1.9 in upward-trending periods, but drops to approximately 0.9 or becomes negative in downward-trending periods.

This is a common issue in hybrid mean-reversion + trend-following systems. In strong 100-day uptrends the two components reinforce each other: trend-following maintains long exposure while mean-reversion buys dips that revert quickly due to bullish momentum. In strong 100-day downtrends the signals conflict — mean-reversion still generates long entries on short-term oversold conditions, while trend-following logic (currently long-only) either stays flat or avoids new positions.

The current ADX filter only detects trend **strength**, not **direction**, so persistent downtrends with high ADX are classified as “trending” and routed to long-biased trend-following rules. Without short-side participation or a directional override, the strategy cannot profit from or fully avoid downside pressure.

Several equity market asymmetries make downtrends particularly challenging:

- Stocks exhibit positive long-term drift, so any longer-window mean carries an upward slope and continues to pull mean-reversion toward longs even in bearish regimes.
- Down moves are accompanied by volatility spikes (leverage effect). Without volatility-adjusted sizing, losses become larger in bear periods.
- Bear markets are shorter but sharper than bull markets, so the 100-day filter sees more historical up-regimes during parameter tuning. Downtrends are rarer and more violent, with higher slippage and wider spreads at hourly frequency.
- Behavioral overreaction tends to be more extreme at the end of bears (panic selling), creating oversold readings that mean-reversion may enter prematurely.

Additional constraints:
- The system is 100% long-only — no short exposure in downtrends.
- Transaction costs and slippage are not modeled.
- Total number of trades is low (18 over ~100 trading days).
- Parameter selection uses a fixed chronological split rather than walk-forward testing.
- Single-instrument testing on SBUX (a survivor with net upward bias) introduces survivorship/optimization bias.

---

## 🚀 Planned Improvements

The following changes are intended to address the limitations:

1. Add short-side logic  
   - mean-reversion shorts when z-score > +2.0 in range-bound regimes  
   - trend-following shorts in downtrending regimes

2. Introduce a longer-term trend filter  
   - 100-day return or price position relative to SMA(200) to adjust bias

3. Expand to multiple instruments  
   - Test across 20–50 stocks or ETFs

4. Improve realism  
   - Include commission and slippage estimates  
   - Implement walk-forward optimization  
   - Add Monte Carlo analysis

5. Explore alternative regime detection  
   - Supervised classification models instead of ADX threshold

6. Refine position sizing  
   - Volatility targeting or fractional Kelly criterion

---

## 🛠️ Tech Stack & Project Structure

- Language: Python 3.9+
- Core libraries: pandas, pandas_ta, numpy, matplotlib, mplfinance, Plotly
- Backtesting: custom event-driven simulator
- Optimization: limited grid search

Directory layout:
- `data_fetcher.py` — data retrieval and cleaning
- `regime_detector.py` — ADX and ATR calculation
- `mean_reversion.py` — z-score based signals
- `trend_following.py` — moving average + volume signals
- `risk_manager.py` — sizing and drawdown rules
- `backtester.py` — simulation engine
- `optimizer.py` — parameter search
- `visualization.py` — plot generation

---

## 📥 How to Run

```bash
# Clone repository
git clone https://github.com/IssacWong0103/Regime-Aware-Algo-Trading-Python-Project.git
cd "Regime-Aware-Algo-Trading-Python-Project/sbux regime trader"

# Install dependencies
pip install -r requirements.txt

# Directly run each lines of code from the testing.ipynb file until the optimization step
testing.ipynb

# Generate graphs for visualization eventually
python visualization.py
