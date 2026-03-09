#!/usr/bin/env python3
"""
Realistic trading simulation with $100 starting capital
Includes fees, slippage, and realistic position sizing
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

print("="*60)
print("💰 $100 REALISTIC TRADING SIMULATION")
print("="*60)

# Load models
old_model_data = joblib.load('data/models/yasen_alpha_champion.joblib')
new_model_data = joblib.load('data/models/yasen_alpha_optimized_fast.joblib')

# Extract models
def extract_model(model_data):
    if isinstance(model_data, dict):
        if 'model' in model_data:
            return model_data['model']
        elif 'models' in model_data:
            return model_data['models'][0]
    return model_data

old_model = extract_model(old_model_data)
new_model = extract_model(new_model_data)

# Load and prepare data (use last 6 months for simulation)
df = pd.read_parquet('data/processed/features_latest.parquet')
df = df.tail(4380).copy()  # Last 6 months of hourly data

exclude_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]

print(f"\n📊 Simulation period: {df.index[0]} to {df.index[-1]}")
print(f"Total hours: {len(df)}")

def simulate_trading(model, initial_capital=100, model_name="Model"):
    capital = initial_capital
    trades = []
    equity_curve = [capital]
    dates = [df.index[0]]
    
    in_position = False
    entry_price = 0
    entry_idx = 0
    
    # Trading parameters
    fee_rate = 0.001  # 0.1% trading fee
    slippage = 0.001  # 0.1% slippage
    risk_per_trade = 0.02  # 2% risk
    
    # Get predictions
    predictions = model.predict_proba(X)[:, 1]
    
    for i in range(len(predictions) - 1):
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i + 1]
        signal = predictions[i] > 0.47
        
        # Exit position if signal changes to HOLD
        if in_position and not signal:
            # Calculate return with fees and slippage
            gross_return = (current_price - entry_price) / entry_price
            net_return = gross_return - fee_rate - slippage
            
            # Position sizing (2% risk)
            risk_amount = capital * risk_per_trade
            position_size = risk_amount / 0.02  # Assuming 2% stop
            pnl = position_size * net_return
            
            capital += pnl
            
            trades.append({
                'entry_date': df.index[entry_idx],
                'exit_date': df.index[i],
                'entry_price': entry_price,
                'exit_price': current_price,
                'gross_return': gross_return,
                'net_return': net_return,
                'pnl': pnl
            })
            
            in_position = False
            equity_curve.append(capital)
            dates.append(df.index[i])
        
        # Enter position on BUY signal
        elif not in_position and signal and predictions[i] > 0.6:
            in_position = True
            entry_price = current_price
            entry_idx = i
    
    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital
    
    if trades:
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
        max_drawdown = min(0, (min(equity_curve) - initial_capital) / initial_capital)
        total_pnl = sum(t['pnl'] for t in trades)
    else:
        win_rate = avg_win = avg_loss = max_drawdown = total_pnl = 0
    
    return {
        'final_capital': capital,
        'total_return': total_return,
        'trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'total_pnl': total_pnl,
        'equity_curve': equity_curve,
        'dates': dates
    }

print("\n📊 Simulating OLD model (59.19%)...")
old_results = simulate_trading(old_model, 100, "Old")

print("📊 Simulating NEW model (64.12%)...")
new_results = simulate_trading(new_model, 100, "New")

print("\n" + "="*70)
print("📈 SIMULATION RESULTS")
print("="*70)
print(f"{'Metric':<20} {'OLD MODEL':<20} {'NEW MODEL':<20} {'DIFFERENCE':<15}")
print("-"*75)

metrics = [
    ('Final Capital', '${:.2f}', '${:.2f}'),
    ('Total Return', '{:.2%}', '{:.2%}'),
    ('Trades', '{:.0f}', '{:.0f}'),
    ('Win Rate', '{:.2%}', '{:.2%}'),
    ('Avg Win', '${:.2f}', '${:.2f}'),
    ('Avg Loss', '${:.2f}', '${:.2f}'),
    ('Max Drawdown', '{:.2%}', '{:.2%}'),
    ('Total P&L', '${:.2f}', '${:.2f}')
]

for metric_name, old_format, new_format in metrics:
    # Convert metric name to dictionary key
    if metric_name == 'Total P&L':
        key = 'total_pnl'
    else:
        key = metric_name.lower().replace(' ', '_')
    
    old_val = old_results[key]
    new_val = new_results[key]
    
    old_str = old_format.format(old_val) if isinstance(old_val, (int, float)) else str(old_val)
    new_str = new_format.format(new_val) if isinstance(new_val, (int, float)) else str(new_val)
    
    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
        diff = new_val - old_val
        if 'Win Rate' in metric_name or 'Return' in metric_name or 'Drawdown' in metric_name:
            diff_str = f'{diff:+.2%}'
        elif 'Capital' in metric_name or 'P&L' in metric_name or 'Win' in metric_name or 'Loss' in metric_name:
            diff_str = f'${diff:+.2f}'
        else:
            diff_str = f'{diff:+.0f}'
    else:
        diff_str = '-'
    
    print(f"{metric_name:<20} {old_str:<20} {new_str:<20} {diff_str:<15}")

print("="*75)

# Recommendation
print("\n✅ RECOMMENDATION:")
if old_results['final_capital'] > new_results['final_capital']:
    print(f"   OLD MODEL WINS! (${old_results['final_capital']:.2f} vs ${new_results['final_capital']:.2f})")
    print(f"   {old_results['trades']} trades with {old_results['win_rate']:.2%} win rate")
    print("   Keep current champion")
else:
    print(f"   NEW MODEL WINS! (${new_results['final_capital']:.2f} vs ${old_results['final_capital']:.2f})")
    print(f"   {new_results['trades']} trades with {new_results['win_rate']:.2%} win rate")

# Print final account values
print("\n" + "="*70)
print(f"💰 FINAL ACCOUNT VALUES")
print("="*70)
print(f"Starting capital: $100.00")
print(f"Old model final:  ${old_results['final_capital']:.2f}")
print(f"New model final:  ${new_results['final_capital']:.2f}")
print(f"Difference:       ${new_results['final_capital'] - old_results['final_capital']:+.2f}")
print("="*70)

# Summary
print("\n📊 SUMMARY:")
print(f"OLD Model: ${old_results['final_capital']:.2f} from {old_results['trades']} trades ({old_results['win_rate']:.2%} win rate)")
print(f"NEW Model: ${new_results['final_capital']:.2f} from {new_results['trades']} trades ({new_results['win_rate']:.2%} win rate)")
print("="*70)
