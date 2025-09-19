
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

@dataclass
class Config:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent

    INPUT_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'processed' / 'eurusd_m5_ml_ready_v2.parquet'
    REPORTS_DIR: Path = _PROJECT_ROOT / 'reports'
    FIGURES_DIR: Path = _PROJECT_ROOT / 'reports' / 'figures'

    TRAIN_PERIOD_LENGTH: str = '6MS'
    TEST_PERIOD_LENGTH: str = '1MS'
    TRANSACTION_COSTS_BPS: float = 1.5
    CONFIDENCE_THRESHOLDS: list = field(default_factory=lambda: [0.95, 0.96])
    PT_MULT: float = 1.2
    SL_MULT: float = 3.0
    TIME_LIMIT_BARS: int = 24
    LGBM_PARAMS: dict = field(default_factory=lambda: {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'n_estimators': 2000, 'learning_rate': 0.02, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'max_depth': 8, 'num_leaves': 63,
        'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42, 'verbose': -1
    })
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

def generate_walk_forward_splits(data: pd.DataFrame, train_period: str, test_period: str):
    """Generates indices for walk-forward training and testing periods."""
    data_start_date = data.index.min()
    data_end_date = data.index.max()
    train_offset = pd.tseries.frequencies.to_offset(train_period)
    test_offset = pd.tseries.frequencies.to_offset(test_period)
    walk_start = data_start_date
    while walk_start + train_offset + test_offset <= data_end_date:
        train_end = walk_start + train_offset
        test_end = train_end + test_offset
        yield walk_start, train_end, test_end
        walk_start += test_offset

def run_event_driven_backtest(pred_probas: pd.DataFrame, test_data: pd.DataFrame, cfg: Config, threshold: float) -> pd.DataFrame:
    """Runs a realistic, event-driven backtest with a given confidence threshold."""
    positions = pd.Series(0, index=test_data.index)
    returns = test_data['returns']
    in_position = False
    sl_level, tp_level, bars_held = 0, 0, 0
    
    for i in range(len(test_data)):
        current_time = test_data.index[i]
        current_high = test_data['high'].iloc[i]
        current_low = test_data['low'].iloc[i]
        
        if in_position:
            bars_held += 1
            position_type = positions.iloc[i-1]
            if (position_type == 1 and (current_low <= sl_level or current_high >= tp_level)) or \
               (position_type == -1 and (current_high >= sl_level or current_low <= tp_level)) or \
               (bars_held >= cfg.TIME_LIMIT_BARS):
                in_position = False
            
            if in_position:
                positions.iloc[i] = position_type
                continue

        if not in_position:
            prob_sl = pred_probas.loc[current_time, 0]
            prob_pt = pred_probas.loc[current_time, 2]
            
            signal = 0
            if prob_pt > threshold: signal = 1
            elif prob_sl > threshold: signal = -1
            
            if signal != 0:
                positions.iloc[i] = signal
                in_position = True
                entry_price = test_data['close'].iloc[i]
                vol = test_data['garch_vol_forecast'].iloc[i]
                if signal == 1:
                    tp_level = entry_price * (1 + vol * cfg.PT_MULT)
                    sl_level = entry_price * (1 - vol * cfg.SL_MULT)
                else:
                    tp_level = entry_price * (1 - vol * cfg.PT_MULT)
                    sl_level = entry_price * (1 + vol * cfg.SL_MULT)
                bars_held = 0

    strategy_returns = positions.shift(1) * returns
    trades = positions.diff().abs()
    transaction_costs = trades * (cfg.TRANSACTION_COSTS_BPS / 10000.0)
    strategy_returns_net = strategy_returns - transaction_costs
    
    return pd.DataFrame({'positions': positions, 'strategy_returns_net': strategy_returns_net})

def calculate_performance_metrics(equity_curve: pd.Series, positions: pd.Series) -> dict:
    """Calculates key performance metrics."""
    total_return = (equity_curve.iloc[-1] - 1) * 100
    daily_returns = equity_curve.resample('D').last().pct_change(fill_method=None)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    total_trades = (positions.diff().abs() > 0).sum()
    
    return {
        'Total Return (%)': total_return, 'Sharpe Ratio (Annualized)': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown, 'Total Trades': total_trades
    }

def main():
    """Main function to orchestrate the threshold optimization pipeline."""
    start_time = time.time()
    cfg = Config()
    logging.info("--- 🚀 Starting Walk-Forward Validation Pipeline (v6 - Threshold Optimization) ---")

    try:
        df_ml = pd.read_parquet(cfg.INPUT_FILEPATH)
        raw_data_dir = cfg.INPUT_FILEPATH.parent.parent / 'raw'
        raw_filepath = list(raw_data_dir.glob('*.parquet'))[0]
        df_raw = pd.read_parquet(raw_filepath)
        df_for_backtest = df_raw[['high', 'low', 'close']].copy()
        df_for_backtest['returns'] = df_for_backtest['close'].pct_change()
        df_for_backtest = df_for_backtest.join(df_ml['garch_vol_forecast'])
        df = df_ml.join(df_for_backtest['returns']).dropna()
        df_for_backtest = df_for_backtest.loc[df.index]
        logging.info(f"✅ Data loaded and prepared. Final dataset has {len(df):,} records.")
    except Exception as e:
        logging.error(f"❌ FATAL ERROR during data loading: {e}", exc_info=True)
        sys.exit(1)

    X = df.drop(columns=['target', 'returns'])
    y = df['target'].map({-1: 0, 0: 1, 1: 2})
    
    all_threshold_results = []
    all_equity_curves = {}
    
    splits = list(generate_walk_forward_splits(df, cfg.TRAIN_PERIOD_LENGTH, cfg.TEST_PERIOD_LENGTH))
    logging.info(f"Generated {len(splits)} walk-forward splits. Pre-training models for all walks...")

    trained_models = []
    for i, (train_start, train_end, test_end) in enumerate(splits):
        train_mask = (X.index >= train_start) & (X.index < train_end)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        model = lgb.LGBMClassifier(**cfg.LGBM_PARAMS)
        model.fit(X_train, y_train)
        trained_models.append(model)
    logging.info("✅ All models trained.")

    for threshold in cfg.CONFIDENCE_THRESHOLDS:
        logging.info(f"\n--- 🔬 Testing Threshold: {threshold:.2f} ---")
        all_walk_results = []
        for i, (train_start, train_end, test_end) in enumerate(splits):
            model = trained_models[i]
            test_mask = (X.index >= train_end) & (X.index < test_end)
            X_test = X.loc[test_mask]
            test_data_slice = df_for_backtest.loc[test_mask]

            if X_test.empty: continue

            y_pred_proba = model.predict_proba(X_test)
            pred_probas_df = pd.DataFrame(y_pred_proba, index=X_test.index, columns=[0, 1, 2])
            
            walk_backtest = run_event_driven_backtest(pred_probas_df, test_data_slice, cfg, threshold)
            all_walk_results.append(walk_backtest)

        full_backtest = pd.concat(all_walk_results)
        full_backtest['equity_curve'] = (1 + full_backtest['strategy_returns_net']).cumprod()
        
        performance_metrics = calculate_performance_metrics(full_backtest['equity_curve'], full_backtest['positions'])
        performance_metrics['Threshold'] = threshold
        all_threshold_results.append(performance_metrics)
        all_equity_curves[threshold] = full_backtest['equity_curve']

    results_df = pd.DataFrame(all_threshold_results).set_index('Threshold')
    logging.info("\n\n" + "="*80)
    logging.info("--- 🏆 Final Threshold Optimization Report 🏆 ---")
    logging.info("="*80)
    print(results_df.to_string(float_format="%.2f"))
    
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = cfg.REPORTS_DIR / 'walk_forward_performance_report_v6_optimization.txt'
    with open(report_path, 'w') as f:
        f.write("Walk-Forward Validation Report (v6 Threshold Optimization)\n")
        f.write("="*80 + "\n")
        f.write(results_df.to_string(float_format="%.2f"))
    logging.info(f"\n💾 Full optimization report saved to {report_path}")

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    for threshold, equity_curve in all_equity_curves.items():
        ax.plot(equity_curve.index, equity_curve, label=f'Threshold = {threshold:.2f}')
    
    ax.set_title('Walk-Forward Validation: Comparative Equity Curves by Threshold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    ax.grid(True)
    plot_path = cfg.FIGURES_DIR / 'walk_forward_equity_curve_v6_optimization.png'
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info(f"💾 Comparative equity curve plot saved to {plot_path}")

    end_time = time.time()
    logging.info(f"\n--- ✅ Pipeline Complete in {end_time - start_time:.2f} seconds ---")
if __name__ == "__main__":
    main()

