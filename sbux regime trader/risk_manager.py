import pandas as pd
from pathlib import Path

class RiskManager:    
    def __init__(self, initial_capital: float = 100_000, risk_per_trade: float = 0.005, max_dd: float = 0.15):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_dd = max_dd
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.paused = False
        print(f"RiskManager initialized: ${initial_capital:,.0f} capital, {risk_per_trade*100:.1f}% per-trade risk")
    
    def update_equity(self, equity: float):
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if dd > self.max_dd:
            self.paused = True
            print(f"Trading PAUSED — max drawdown {dd*100:.1f}% exceeded")
    
    # Volatility-based sizing --> return 0 if ATR invalid
    def calculate_position_size(self, atr: float, price: float) -> int:
        if pd.isna(atr) or atr <= 0 or self.paused:
            return 0
        
        risk_amount = self.current_equity * self.risk_per_trade
        stop_distance = atr * 1.5  # conservative buffer
        shares = int(risk_amount / stop_distance)
        return max(1, shares)  # at least 1 share if risk allows
    
    def volatility_scale(self, current_atr: float, avg_atr: float) -> float:
        if pd.isna(current_atr) or pd.isna(avg_atr) or current_atr <= 0:
            return 1.0  # neutral if no vol info
        if current_atr > avg_atr * 1.5:
            return 0.5
        return 1.0
    
    def can_trade(self) -> bool:
        return not self.paused