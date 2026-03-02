import pandas as pd
from pathlib import Path
from backtester import load_all_data, run_backtest, calculate_metrics

# Limited optimization to prevent curve-fitting
def optimize_parameters():
    df_full = load_all_data()
    split_idx = int(len(df_full) * 0.7)
    df_is = df_full.iloc[:split_idx].copy()
    
    # Clean any lingering MultiIndex/duplicate signal columns
    for col in ['mr_signal', 'mr_exit', 'mr_position', 'tf_signal', 'tf_exit', 'tf_position', 'trailing_stop']:
        if col in df_is.columns:
            df_is = df_is.drop(columns=[col])
    
    param_grid = [
        {"z_entry": 1.8, "fast": 8, "slow": 40, "atr_mult": 2.5},
        {"z_entry": 2.0, "fast": 10, "slow": 50, "atr_mult": 3.0},   # baseline
        {"z_entry": 2.2, "fast": 12, "slow": 60, "atr_mult": 3.5},
        {"z_entry": 1.9, "fast": 9, "slow": 45, "atr_mult": 2.8},
        {"z_entry": 2.1, "fast": 11, "slow": 55, "atr_mult": 3.2},
    ]
    
    best_sharpe = -999
    best_params = None
    best_results = None
    
    print("Starting limited in-sample optimization (5 combos only)...\n")
    
    from mean_reversion import mean_reversion_logic
    from trend_following import trend_following_logic
    
    for params in param_grid:
        # Generate fresh signals for this parameter set on in-sample data only
        df_mr = mean_reversion_logic(df_is, z_entry=params["z_entry"])
        df_tf = trend_following_logic(df_is, fast=params["fast"], slow=params["slow"], atr_mult=params["atr_mult"])
        
        # Join fresh signals
        df_opt = df_is.copy()
        df_opt = df_opt.join(df_mr[['mr_signal', 'mr_exit', 'mr_position']])
        df_opt = df_opt.join(df_tf[['tf_signal', 'tf_exit', 'tf_position', 'trailing_stop']])
        
        # Run backtest on this parameter set, in-sample only
        df_bt = run_backtest(df_opt, initial_capital=100_000)
        
        metrics = calculate_metrics(df_bt, period_name=f"IS - z={params['z_entry']}, MA={params['fast']}/{params['slow']}, ATRx{params['atr_mult']}")
        
        if metrics["sharpe_daily"] > best_sharpe:
            best_sharpe = metrics["sharpe_daily"]
            best_params = params
            best_results = metrics
    
    print(f"\nBest parameters (in-sample only — no OOS tuning):")
    print(f"   z_entry       = {best_params['z_entry']}")
    print(f"   MA fast/slow  = {best_params['fast']}/{best_params['slow']}")
    print(f"   ATR multiplier = {best_params['atr_mult']}")
    print(f"   Best IS Sharpe = {best_sharpe:.2f}")
    
    # Save for final run
    pd.Series(best_params).to_csv("results/best_params.csv")
    print("Best params saved → results/best_params.csv")
    
    return best_params

if __name__ == "__main__":
    best = optimize_parameters()