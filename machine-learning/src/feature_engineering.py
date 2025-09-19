import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numba
import pandas as pd
from arch import arch_model
import pandas_ta as ta
from saxpy.sax import sax_via_window

@dataclass
class Config:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    
    INPUT_FILENAME: str = "EURUSD_M5_2024-01-01_2025-07-25.parquet"
    INPUT_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'raw' / INPUT_FILENAME
    OUTPUT_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'processed' / 'eurusd_m5_ml_ready.parquet'

    GARCH_WINDOW: int = 1000
    GARCH_UPDATE_FREQ: int = 250
    VOL_ROLLING_WINDOW: int = 20

    PT_MULT: float = 2.0
    SL_MULT: float = 2.0
    TIME_LIMIT: int = 24  
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def create_sax_features(df: pd.DataFrame, window_size: int = 24, alphabet_size: int = 5) -> pd.DataFrame:
    logging.info(f"Engineering SAX features (window={window_size}, alphabet={alphabet_size})...")
    
    rolling_mean = df['close'].rolling(window=window_size).mean()
    rolling_std = df['close'].rolling(window=window_size).std()
    normalized_price = (df['close'] - rolling_mean) / rolling_std
    normalized_price.dropna(inplace=True)

    sax_words = sax_via_window(normalized_price, win_size=window_size, alpha_size=alphabet_size, squeeze_symbolic_representation=True)
    sax_series = pd.Series(sax_words)
    sax_dummies = pd.get_dummies(sax_series)
    lookback_window = 144
    motif_counts = sax_dummies.rolling(window=lookback_window, min_periods=1).sum()
    motif_counts = motif_counts.add_prefix('sax_')
    
    df = df.join(motif_counts)
    
    logging.info(f"Added {len(motif_counts.columns)} SAX motif features.")
    return df

def create_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers macro-context features from the 15-minute timeframe.
    """
    logging.info("Engineering multi-timeframe (15M) context features...")

    agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    df_15m = df.resample('15T').agg(agg_rules).dropna()
    
    df_15m.ta.rsi(length=14, append=True, col_names=('rsi_15m',))
    df_15m.ta.atr(length=14, append=True, col_names=('atr_15m',))
    rolling_15m = df_15m['close'].rolling(window=14)
    rolling_max_15m = rolling_15m.max()
    rolling_min_15m = rolling_15m.min()
    df_15m['price_pos_15m'] = (df_15m['close'] - rolling_min_15m) / (rolling_max_15m - rolling_min_15m)
    
    features_15m = df_15m[['rsi_15m', 'atr_15m', 'price_pos_15m']]
    
    df = df.join(features_15m)
    df[['rsi_15m', 'atr_15m', 'price_pos_15m']] = df[['rsi_15m', 'atr_15m', 'price_pos_15m']].fillna(method='ffill')
    
    logging.info("Multi-timeframe features added.")
    return df

def calculate_rolling_garch(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Calculates GARCH(1,1) volatility on a rolling window with periodic refitting.
    """
    logging.info(f"Calculating rolling GARCH({config.GARCH_WINDOW}-bar window, {config.GARCH_UPDATE_FREQ}-bar refit)...")
    
    returns_for_garch = df['returns'].dropna() * 100
    garch_forecasts = pd.Series(index=df.index, dtype=float)
    last_res = None

    for i in range(config.GARCH_WINDOW, len(returns_for_garch)):
        current_timestamp = returns_for_garch.index[i]
        if (i - config.GARCH_WINDOW) % config.GARCH_UPDATE_FREQ == 0:
            current_window = returns_for_garch.iloc[i - config.GARCH_WINDOW : i]
            model = arch_model(current_window, p=1, q=1, vol='Garch', dist='Normal', rescale=False)
            try:
                last_res = model.fit(disp='off', show_warning=False)
            except Exception:
                last_res = None 
        if last_res and last_res.convergence_flag == 0:
            forecast = last_res.forecast(horizon=1, reindex=False)
            garch_forecasts.loc[current_timestamp] = np.sqrt(forecast.variance.iloc[-1, 0])

    df['garch_vol_forecast'] = garch_forecasts / 100
    logging.info("GARCH calculation complete.")
    return df

def create_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Orchestrates the creation of all predictive features."""
    logging.info("Engineering feature set...")
    
    df['returns'] = df['close'].pct_change()

    lags = [1, 2, 3, 5, 10, 20, 30, 60]
    for lag in lags:
        df[f'log_return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))

    df['return_skew_20'] = df['returns'].rolling(20).skew()
    df['return_kurt_20'] = df['returns'].rolling(20).kurt()

    df = calculate_rolling_garch(df, config)

    df['price_acceleration'] = df['returns'].diff()
    df['return_autocorr_5'] = df['returns'].rolling(config.VOL_ROLLING_WINDOW).apply(lambda x: x.autocorr(lag=5), raw=False)

    df['spread_proxy'] = (df['high'] - df['low']) / df['close']

    df['vol_regime'] = (df['garch_vol_forecast'] > df['garch_vol_forecast'].rolling(100).quantile(0.8)).astype(int)

    df['vol_momentum'] = df['garch_vol_forecast'].pct_change(5)

    rolling_max = df['high'].rolling(config.VOL_ROLLING_WINDOW).max()
    rolling_min = df['low'].rolling(config.VOL_ROLLING_WINDOW).min()
    df['dist_to_high'] = (df['close'] - rolling_max) / df['close']
    df['dist_to_low'] = (df['close'] - rolling_min) / df['close']

    df['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)

    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df = create_multi_timeframe_features(df)

    df = create_sax_features(df) 
    logging.info(f"Feature engineering complete.")
    return df

@numba.jit(nopython=True, cache=True)
def calculate_target_labels_numba(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volatility: np.ndarray, pt_mult: float, sl_mult: float, time_limit: int
) -> np.ndarray:

    n = len(close)
    target = np.full(n, 0, dtype=np.int64)

    for i in range(n - time_limit -1):
        if np.isnan(volatility[i]) or volatility[i] <= 1e-9:
            target[i] = np.nan
            continue

        entry_price = close[i]
        upper_barrier = entry_price * (1 + (volatility[i] * pt_mult))
        lower_barrier = entry_price * (1 - (volatility[i] * sl_mult))

        for j in range(1, time_limit + 1):
            future_high = high[i + j]
            future_low = low[i + j]
            hit_upper = future_high >= upper_barrier
            hit_lower = future_low <= lower_barrier

            if hit_upper and hit_lower:
                target[i] = 0 
                break 
            elif hit_upper:
                target[i] = 1
                break
            elif hit_lower:
                target[i] = -1
                break
    
    return target

def generate_labels(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    logging.info("Defining target variable with Triple-Barrier Method...")
    
    cols_for_labeling = ['close', 'high', 'low', 'garch_vol_forecast']
    df_for_labeling = df[cols_for_labeling].dropna()

    targets = calculate_target_labels_numba(
        close=df_for_labeling['close'].values,
        high=df_for_labeling['high'].values,
        low=df_for_labeling['low'].values,
        volatility=df_for_labeling['garch_vol_forecast'].values,
        pt_mult=config.PT_MULT,
        sl_mult=config.SL_MULT,
        time_limit=config.TIME_LIMIT
    )
    
    df['target'] = pd.Series(targets, index=df_for_labeling.index)
    logging.info("Labeling complete.")
    return df

def main():
    """Main function to orchestrate the feature engineering pipeline."""
    start_time = time.time()
    cfg = Config()
    logging.info("--- Starting EUR/USD Advanced Feature Engineering Pipeline ---")

    try:
        df = pd.read_parquet(cfg.INPUT_FILEPATH)
        if 'volume' in df.columns:
            df.drop(columns=['volume'], inplace=True)
        logging.info(f"Loaded {len(df):,} M5 records from {cfg.INPUT_FILEPATH}")
    except FileNotFoundError:
        logging.error(f"FATAL ERROR: Data file not found at {cfg.INPUT_FILEPATH}")
        sys.exit(1)
    df = create_features(df, cfg)
    df = generate_labels(df, cfg)

    logging.info("Cleaning and saving final dataset...")
    feature_columns = [col for col in df.columns if col not in [
        'open', 'high', 'low', 'close', 'returns', 'target'
    ]]
    
    final_df = df[feature_columns + ['target']].copy()
    final_df.dropna(inplace=True)
    final_df['target'] = final_df['target'].astype(int)

    logging.info(f"Final ML-Ready Dataset Shape: {final_df.shape}")
    logging.info("Final Target Distribution:")
    print(final_df['target'].value_counts(normalize=True).sort_index().to_string())
    cfg.OUTPUT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(cfg.OUTPUT_FILEPATH, engine='pyarrow', compression='snappy')
    logging.info(f"Successfully saved dataset to: {cfg.OUTPUT_FILEPATH}")

    end_time = time.time()
    logging.info(f"--- Pipeline Complete in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
