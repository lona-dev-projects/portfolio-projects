
# Implement new features:
# - Rolling 6M train / 1M test windows
# - Model regularization testing
# - Event-driven realistic backtesting
# - Portfolio management with compounding 
# - Continuous parameter optimization
# - Multi-objective scoring with AUC diagnostics
# - Efficient pre-computation and caching
import time
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.optimize import differential_evolution
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class UltimateConfig:
    """Configuration for the V4 Walk-Forward Validation."""
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent

    INPUT_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'processed' / 'eurusd_m5_ml_ready_v4.parquet'
    RAW_DATA_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'raw' / 'EURUSD_M5_2024-01-01_2025-07-25.parquet'
    REPORTS_DIR: Path = _PROJECT_ROOT / 'reports'
    FIGURES_DIR: Path = _PROJECT_ROOT / 'reports' / 'figures'

    TRAIN_PERIOD_LENGTH: str = '5MS'
    VALIDATION_PERIOD_LENGTH: str = '1MS'
    TEST_PERIOD_LENGTH: str = '1MS' 

    INITIAL_CAPITAL: float = 100.0
    RISK_PER_TRADE_PCT: float = 1.0
    TRANSACTION_COSTS_BPS: float = 1.5 
    SPREAD_BPS: float = 1.0 
    MODEL_CONFIGS: Dict[str, Dict] = field(default_factory=lambda: {
        'v4_model': {
            'objective': 'multiclass', 
            'num_class': 3, 
            'metric': 'multi_logloss',
            'n_estimators': 500,  
            'learning_rate': 0.05,  
            'feature_fraction': 0.8,  
            'bagging_fraction': 0.8,  
            'max_depth': 7,  
            'num_leaves': 40,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'class_weight': 'balanced', 
            'n_jobs': -1, 
            'random_state': 42, 
            'verbose': -1
        }
    })

    PARAM_BOUNDS: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.70, 0.95), 
        (1.0, 2.5),   
        (1.5, 3.5),  
    ])
    
    TIME_LIMIT_BARS: int = 24 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

class UltimatePortfolioManager:
    """Manages portfolio state, trades, and performance metrics."""
    
    def __init__(self, initial_cash: float, spread_bps: float, cost_bps: float, risk_pct: float, time_limit_bars: int):
        self.initial_cash = initial_cash
        self.spread = spread_bps / 10000.0
        self.cost_per_trade = cost_bps / 10000.0
        self.risk_per_trade = risk_pct / 100.0
        self.time_limit_bars = time_limit_bars
        self.cash = initial_cash
        self.equity = initial_cash
        self.position = None
        self.peak_equity = initial_cash
        self.max_drawdown = 0.0
        
        self.trade_log = []
        self.equity_curve = []
        self.timestamps = []
        
    def update_equity(self, timestamp: pd.Timestamp, high: float, low: float, close: float) -> bool:
        """Update equity and check for exit conditions. Returns True if position closed."""
        self.timestamps.append(timestamp)
        
        unrealized_pnl = 0.0
        if self.position:
            if self.position['type'] == 'BUY':
                unrealized_pnl = (close - self.position['entry_price']) * self.position['size']
            else:  
                unrealized_pnl = (self.position['entry_price'] - close) * self.position['size']
        
        self.equity = self.cash + unrealized_pnl
        self.equity_curve.append(self.equity)
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_dd)
        
        position_closed = False
        if self.position:
            position_closed = self._check_exit_conditions(timestamp, high, low, close)
            
        return position_closed

    def execute_signal(self, signal: str, timestamp: pd.Timestamp, price: float, 
                       sl_price: float, tp_price: float) -> bool:
        """Execute trading signal with risk controls."""
        if signal not in ['BUY', 'SELL'] or self.position:
            return False
        
        risk_per_unit = abs(price - sl_price)
        if risk_per_unit <= 1e-9:
            return False
        
        cash_to_risk = self.equity * self.risk_per_trade
        position_size = cash_to_risk / risk_per_unit
        
        entry_price = price + (self.spread / 2) if signal == 'BUY' else price - (self.spread / 2)
        
        self.position = {
            'type': signal,
            'entry_price': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'size': position_size,
            'entry_time': timestamp,
            'entry_equity': self.equity,
            'bars_held': 0
        }
        return True
    
    def _check_exit_conditions(self, timestamp: pd.Timestamp, high: float, low: float, close: float) -> bool:
        """Check if position should be closed based on SL, TP, or time limit."""
        if not self.position:
            return False
            
        self.position['bars_held'] += 1
        
        exit_price, exit_reason = None, None
        sl, tp = self.position['sl'], self.position['tp']
        
        if self.position['type'] == 'BUY':
            if low <= sl: exit_price, exit_reason = sl, "STOP_LOSS"
            elif high >= tp: exit_price, exit_reason = tp, "TAKE_PROFIT"
        else: # SELL
            if high >= sl: exit_price, exit_reason = sl, "STOP_LOSS"
            elif low <= tp: exit_price, exit_reason = tp, "TAKE_PROFIT"

        if not exit_price and self.position['bars_held'] >= self.time_limit_bars:
            exit_price, exit_reason = close, "TIME_LIMIT"
            
        if exit_price:
            self._close_position(timestamp, exit_price, exit_reason)
            return True
            
        return False
    
    def _close_position(self, timestamp: pd.Timestamp, price: float, reason: str):
        """Close the current position and log the trade."""
        if not self.position:
            return
            
        exit_price = price - (self.spread / 2) if self.position['type'] == 'BUY' else price + (self.spread / 2)
        
        if self.position['type'] == 'BUY':
            gross_pnl = (exit_price - self.position['entry_price']) * self.position['size']
        else:
            gross_pnl = (self.position['entry_price'] - exit_price) * self.position['size']
            
        transaction_cost = self.position['size'] * self.position['entry_price'] * self.cost_per_trade
        net_pnl = gross_pnl - transaction_cost
        
        self.cash += net_pnl
        
        self.trade_log.append({
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'type': self.position['type'],
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'net_pnl': net_pnl,
            'pnl_pct': (net_pnl / self.position['entry_equity']) * 100 if self.position['entry_equity'] > 0 else 0,
            'reason': reason,
        })
        
        self.position = None
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for the backtest."""
        if not self.trade_log:
            return {'total_trades': 0}
            
        trades_df = pd.DataFrame(self.trade_log)
        total_trades = len(trades_df)
        
        total_return = (self.equity / self.initial_cash - 1) * 100
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = winning_trades['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': self.max_drawdown * 100,
            'final_equity': self.equity
        }

class UltimateMLStrategy:
    def __init__(self, confidence_threshold: float, sl_mult: float, tp_mult: float):
        self.confidence_threshold = confidence_threshold
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
    
    def generate_signal(self, data_row: pd.Series) -> Tuple[str, Optional[float], Optional[float]]:
        """Generate a BUY, SELL, or HOLD signal."""
        prob_sell, prob_buy = data_row['prob_sell'], data_row['prob_buy']
        price = data_row['close']
        volatility = data_row.get('garch_vol_forecast', 0)
        
        if pd.isna(volatility) or volatility <= 1e-9:
            return 'HOLD', None, None
            
        if prob_buy > self.confidence_threshold:
            tp_price = price + (price * volatility * self.tp_mult)
            sl_price = price - (price * volatility * self.sl_mult)
            return 'BUY', sl_price, tp_price
        elif prob_sell > self.confidence_threshold:
            tp_price = price - (price * volatility * self.tp_mult)
            sl_price = price + (price * volatility * self.sl_mult)
            return 'SELL', sl_price, tp_price
            
        return 'HOLD', None, None

def generate_walk_forward_splits(data: pd.DataFrame, train_period: str, 
                                 validation_period: str, test_period: str):
    """Generate rolling walk-forward splits for Train, Validation, and Test sets."""
    data_start, data_end = data.index.min(), data.index.max()
    
    train_offset = pd.tseries.frequencies.to_offset(train_period)
    validation_offset = pd.tseries.frequencies.to_offset(validation_period)
    test_offset = pd.tseries.frequencies.to_offset(test_period)
    
    walk_start = data_start
    while walk_start + train_offset + validation_offset + test_offset <= data_end:
        train_end = walk_start + train_offset
        validation_end = train_end + validation_offset
        test_end = validation_end + test_offset
        
        yield walk_start, train_end, validation_end, test_end
        
        walk_start += test_offset

def multi_objective_score(metrics: Dict[str, float], auc_score: float) -> float:
    """Multi-objective scoring function combining trading and ML metrics."""
    if metrics.get('total_trades', 0) < 5: 
        return -1000
    
    return_score = max(0, metrics['total_return_pct']) / 100
    pf_score = min(metrics.get('profit_factor', 0), 5) / 5
    dd_score = max(0, 100 - metrics['max_drawdown_pct']) / 100
    auc_score_norm = max(0, (auc_score - 0.5) * 2)
    
    return (return_score * 0.4 + pf_score * 0.3 + dd_score * 0.2 + auc_score_norm * 0.1)

def objective_function(params: List[float], model: Any, data_to_optimize: pd.DataFrame, 
                       config: UltimateConfig) -> float:
    """Objective function for parameter optimization."""
    confidence_threshold, sl_mult, tp_mult = params
    
    try:
        strategy = UltimateMLStrategy(confidence_threshold, sl_mult, tp_mult)
        portfolio = UltimatePortfolioManager(
            config.INITIAL_CAPITAL, config.SPREAD_BPS, config.TRANSACTION_COSTS_BPS, 
            config.RISK_PER_TRADE_PCT, config.TIME_LIMIT_BARS
        )
        
        X_features = data_to_optimize[model.feature_name_]
        pred_proba = model.predict_proba(X_features) 
        proba_cols = ['prob_sell', 'prob_hold', 'prob_buy']
        df_proba = pd.DataFrame(pred_proba, index=X_features.index, columns=proba_cols)
        local_data = data_to_optimize.join(df_proba)
        
        for i in range(len(local_data)):
            row_data = local_data.iloc[i]
            portfolio.update_equity(row_data.name, row_data['high'], row_data['low'], row_data['close'])
            if not portfolio.position:
                signal, sl, tp = strategy.generate_signal(row_data)
                if signal in ['BUY', 'SELL']:
                    portfolio.execute_signal(signal, row_data.name, row_data['close'], sl, tp)
        
        metrics = portfolio.get_performance_metrics()
        
        y_labels = data_to_optimize['target'].map({-1: 0, 0: 1, 1: 2})
        auc_score = roc_auc_score(y_labels, pred_proba, multi_class='ovr') if len(y_labels.unique()) > 1 else 0.5
        
        return -multi_objective_score(metrics, auc_score)

    except Exception as e:
        logging.warning(f"Optimization error: {e}")
        return 1000

def optimize_parameters(model: Any, validation_data: pd.DataFrame, config: UltimateConfig) -> Tuple[Dict[str, float], float]:
    """Optimize strategy parameters using differential evolution."""
    result = differential_evolution(
        objective_function,
        config.PARAM_BOUNDS,
        args=(model, validation_data, config),
        maxiter=25, 
        popsize=15, 
        seed=42,
        polish=True,
    )
    
    best_params = {
        'confidence_threshold': result.x[0],
        'sl_mult': result.x[1], 
        'tp_mult': result.x[2]
    }
    return best_params, -result.fun
def main():
    """Ultimate walk-forward validation orchestrator for V4."""
    start_time = time.time()
    config = UltimateConfig()
    
    logging.info("🚀 Starting V4 Walk-Forward Validation System")
    
    try:
        df_ml = pd.read_parquet(config.INPUT_FILEPATH)
        df_raw = pd.read_parquet(config.RAW_DATA_FILEPATH)
        logging.info(f"✅ Loaded {len(df_ml):,} V4 feature records and {len(df_raw):,} raw records.")
        df_combined = df_ml.join(df_raw[['high', 'low', 'close']], how='inner').dropna(subset=['target'])
        logging.info(f"✅ Combined dataset ready: {len(df_combined):,} records")
        
    except Exception as e:
        logging.error(f"❌ Data loading failed: {e}")
        sys.exit(1)
    if 'sax_motif' in df_combined.columns:
        df_combined['sax_motif'] = df_combined['sax_motif'].astype('category')

    X = df_combined.drop(columns=['target', 'time_to_hit', 'high', 'low', 'close'])
    y = df_combined['target'].map({-1: 0, 0: 1, 1: 2})
    
    splits = list(generate_walk_forward_splits(
        df_combined, config.TRAIN_PERIOD_LENGTH, config.VALIDATION_PERIOD_LENGTH, config.TEST_PERIOD_LENGTH
    ))
    logging.info(f"🔄 Generated {len(splits)} walk-forward splits")
    
    all_walk_results = []

    for walk_idx, (train_start, train_end, validation_end, test_end) in enumerate(splits):
        logging.info(f"\n--- 🏃‍♂️ Walk {walk_idx + 1}/{len(splits)} [{train_start.date()} to {test_end.date()}] ---")
        
        train_mask = (X.index >= train_start) & (X.index < train_end)
        validation_mask = (X.index >= train_end) & (X.index < validation_end)
        test_mask = (X.index >= validation_end) & (X.index < test_end)
        
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_val, y_val = X.loc[validation_mask], y.loc[validation_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        
        validation_data = df_combined.loc[validation_mask]
        test_data = df_combined.loc[test_mask]
        
        if X_train.empty or X_val.empty or X_test.empty:
            logging.warning("⚠️ Empty data slice, skipping walk")
            continue
        
        #Handle categorical feature
        categorical_features = ['sax_motif'] if 'sax_motif' in X_train.columns else []

        logging.info(f"🎓 Training model on {len(X_train):,} samples...")
        model = lgb.LGBMClassifier(**config.MODEL_CONFIGS['v4_model'])
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            categorical_feature=categorical_features
        )
        
        logging.info("🔧 Optimizing trading parameters on VALIDATION set...")
        best_params, best_score = optimize_parameters(model, validation_data, config)
        logging.info(f"🎯 Best params: {best_params} (Score: {best_score:.4f})")
        
        logging.info("📈 Backtesting on out-of-sample TEST set...")
        strategy = UltimateMLStrategy(**best_params)
        y_pred_proba_test = model.predict_proba(X_test)
        df_proba = pd.DataFrame(y_pred_proba_test, index=X_test.index, columns=['prob_sell', 'prob_hold', 'prob_buy'])
        test_data_with_preds = test_data.join(df_proba)

        portfolio = UltimatePortfolioManager(
            config.INITIAL_CAPITAL, config.SPREAD_BPS, config.TRANSACTION_COSTS_BPS,
            config.RISK_PER_TRADE_PCT, config.TIME_LIMIT_BARS
        )
        
        for i in range(len(test_data_with_preds)):
            row = test_data_with_preds.iloc[i]
            portfolio.update_equity(row.name, row['high'], row['low'], row['close'])
            if not portfolio.position:
                signal, sl, tp = strategy.generate_signal(row)
                if signal in ['BUY', 'SELL']:
                    portfolio.execute_signal(signal, row.name, row['close'], sl, tp)
        
        metrics = portfolio.get_performance_metrics()
        walk_return_pct = metrics.get('total_return_pct', 0)
        logging.info(f"💰 Walk Return: {walk_return_pct:.2f}% (${portfolio.initial_cash:.0f} → ${portfolio.equity:.2f})")

        all_walk_results.append({
            'walk': walk_idx + 1,
            'period': f"{validation_end.strftime('%Y-%m')}",
            **metrics
        })

    logging.info(f"\n{'='*80}")
    logging.info("🏆 V4 VALIDATION RESULTS 🏆")
    logging.info(f"{'='*80}")
    
    results_df = pd.DataFrame(all_walk_results)
    
    if not results_df.empty:
        profitable_walks = results_df[results_df['total_return_pct'] > 0]
        win_rate_walks = len(profitable_walks) / len(results_df) * 100
        
        logging.info(f"🔄 Walk Win Rate: {win_rate_walks:.1f}% ({len(profitable_walks)}/{len(results_df)} profitable)")
        logging.info(f"💰 Average Return per Walk: {results_df['total_return_pct'].mean():.2f}%")
        logging.info(f"📈 Best Walk: {results_df['total_return_pct'].max():.2f}%")
        logging.info(f"📉 Worst Walk: {results_df['total_return_pct'].min():.2f}%")
        
        report_path = config.REPORTS_DIR / 'v4_validation_summary.csv'
        results_df.to_csv(report_path, index=False)
        logging.info(f"💾 Detailed results saved to: {report_path}")

        # Plotting results
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        results_df['total_return_pct'].plot(kind='bar', ax=ax, 
                                            color=results_df['total_return_pct'].apply(lambda x: 'g' if x > 0 else 'r'),
                                            alpha=0.7)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title('V4 Walk-Forward Performance: Return per Walk', fontsize=16)
        ax.set_ylabel('Return (%)')
        ax.set_xlabel('Walk-Forward Period')
        plt.tight_layout()
        
        plot_path = config.FIGURES_DIR / 'v4_walk_forward_returns.png'
        plt.savefig(plot_path, dpi=300)
        logging.info(f"🖼️ Performance plot saved to: {plot_path}")

    end_time = time.time()
    logging.info(f"\n--- ✅ V4 Validation Finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
