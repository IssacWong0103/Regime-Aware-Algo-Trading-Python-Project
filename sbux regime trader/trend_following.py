import pandas as pd
from pathlib import Path

def load_regime_data() -> pd.DataFrame:
    path = Path("data/sbux_1h_with_regime.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded regime-enhanced SBUX 1H data: {df.shape[0]:,} bars")
    return df

# Trend following ONLY in trending regimes + ATR trailing stop
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

if __name__ == "__main__":
    df = load_regime_data()
    df_tf = trend_following_logic(df, fast=10, slow=50, atr_mult=3.0)
    
    df_tf.to_parquet("data/sbux_1h_tf_signals.parquet", compression="gzip")
    print("Trend Following signals + trailing stops saved → data/sbux_1h_tf_signals.parquet")
    
    print("\nLast 15 bars with TF signals:")
    print(df_tf[['Close', 'ma_fast', 'ma_slow', 'regime', 'tf_signal', 'tf_exit', 'tf_position']].tail(15))