import pandas as pd
from pathlib import Path

def load_regime_data() -> pd.DataFrame:
    path = Path("data/sbux_1h_with_regime.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded regime-enhanced SBUX 1H data: {df.shape[0]:,} bars")
    return df

# Short-term z-score (20-bar window — short-term focus)
def calculate_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df['mean'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std(ddof=0)
    df['zscore'] = (df['Close'] - df['mean']) / df['std']
    return df

# Mean Reversion ONLY in range_bound regimes
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

if __name__ == "__main__":
    df = load_regime_data()
    df_mr = mean_reversion_logic(df, z_entry=2.0, z_stop=3.0)
    
    df_mr.to_parquet("data/sbux_1h_mr_signals.parquet", compression="gzip")
    print("Mean Reversion signals + positions saved → data/sbux_1h_mr_signals.parquet")
    
    print("\nLast 15 bars with signals/positions:")
    print(df_mr[['Close', 'zscore', 'regime', 'mr_signal', 'mr_exit', 'mr_position']].tail(15))