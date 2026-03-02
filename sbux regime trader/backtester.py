import pandas as pd
from pathlib import Path
from risk_manager import RiskManager
import numpy as np

def load_all_data() -> pd.DataFrame:
    """Merge regime + MR + TF signals (single-asset)."""
    regime_path = Path("data/sbux_1h_with_regime.parquet")
    mr_path = Path("data/sbux_1h_mr_signals.parquet")
    tf_path = Path("data/sbux_1h_tf_signals.parquet")
    
    df = pd.read_parquet(regime_path)
    mr = pd.read_parquet(mr_path)[['mr_signal', 'mr_exit', 'mr_position']]
    tf = pd.read_parquet(tf_path)[['tf_signal', 'tf_exit', 'tf_position', 'trailing_stop']]
    
    df = df.join(mr).join(tf)
    print(f"Merged full dataset: {df.shape[0]:,} bars")
    return df

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

def calculate_metrics(df: pd.DataFrame, period_name: str = "Full Period", rf_annual: float = 0.04) -> dict:
    if 'equity' not in df.columns or len(df) < 2:
        print(f"Warning: Not enough data for {period_name}")
        return {"total_return": 0.0, "sharpe_daily": 0.0, "max_dd": 0.0}

    total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
    equity_series = df['equity']
    max_dd = ((equity_series / equity_series.cummax()) - 1).min() * 100
    trades_mask = df['trade_return'] != 0
    n_trades = trades_mask.sum()
    if n_trades > 0:
        win_rate = (df['trade_return'][trades_mask] > 0).mean() * 100
        gross_profit = df['trade_return'][df['trade_return'] > 0].sum()
        gross_loss   = abs(df['trade_return'][df['trade_return'] < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = float('inf')

    # Resample equity to daily (last value of each calendar day)
    daily_equity = equity_series.resample('D').last().ffill().dropna()

    if len(daily_equity) < 10:  # too few days → unreliable
        sharpe_daily = 0.0
    else:
        daily_returns = daily_equity.pct_change().dropna()
        if daily_returns.std() < 1e-10:
            sharpe_daily = 0.0
        else:
            rf_daily = rf_annual / 252
            excess_mean = daily_returns.mean() - rf_daily
            sharpe_daily = excess_mean / daily_returns.std() * np.sqrt(252)

    print(f"\n{period_name} Metrics (SBUX 1H)")
    print(f"   Total Return:      {total_return:8.1f}%")
    print(f"   Sharpe (daily)     {sharpe_daily:8.2f}   ← annualized using daily returns")
    print(f"   Max Drawdown:      {max_dd:8.1f}%")
    print(f"   Win Rate:          {win_rate:8.1f}%")
    print(f"   Profit Factor:     {profit_factor:8.2f}")
    print(f"   Number of trades:  {n_trades:8d}")

    return {
        "total_return": total_return,
        "sharpe_daily": sharpe_daily,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades
    }

if __name__ == "__main__":
    df = load_all_data()
    df_bt = run_backtest(df, initial_capital=100_000)
    
    # 70/30 split (in-sample tuning, out-of-sample validation)
    split_idx = int(len(df_bt) * 0.7)
    df_is = df_bt.iloc[:split_idx]
    df_oos = df_bt.iloc[split_idx:]
    
    print("\n=== IN-SAMPLE (70%) ===")
    calculate_metrics(df_is)
    print("\n=== OUT-OF-SAMPLE (30%) ===")
    calculate_metrics(df_oos)
    
    df_bt.to_parquet("data/sbux_1h_backtest_results.parquet", compression="gzip")
    print("Full backtest saved → data/sbux_1h_backtest_results.parquet")