import time
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import pandas as pd
import numpy as np
import numba
from arch import arch_model
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize

warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class Config:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent

    INPUT_FILENAME: str = "EURUSD_M5_2024-01-01_2025-07-25.parquet"
    RAW_DATA_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'raw' / INPUT_FILENAME
    OUTPUT_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'processed' / 'eurusd_m5_ml_ready_v4.parquet'
    GARCH_WINDOW: int = 1000
    GARCH_UPDATE_FREQ: int = 250
    VOL_ROLLING_WINDOW: int = 20
    LONG_TERM_WINDOW: int = 144  
    SAX_WINDOW_SIZE: int = 24    
    SAX_ALPHABET_SIZE: int = 5
    FRACTAL_DIMENSION_WINDOW: int = 48
    PT_MULT: float = 2.0
    SL_MULT: float = 2.0
    TIME_LIMIT: int = 24

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

def timer_decorator(func):
    """Decorator to time and log the execution of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"'{func.__name__}' completed in {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

@timer_decorator
def calculate_rolling_garch_optimized(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    logging.info("Calculating optimized rolling GARCH volatility...")
    
    returns_for_garch = df['returns'].dropna() * 100
    garch_forecasts = pd.Series(index=df.index, dtype=float)
    last_res = None

    for i in range(config.GARCH_WINDOW, len(returns_for_garch)):
        current_timestamp = returns_for_garch.index[i]
        if (i - config.GARCH_WINDOW) % config.GARCH_UPDATE_FREQ == 0:
            current_window = returns_for_garch.iloc[i - config.GARCH_WINDOW : i]
            try:
                model = arch_model(current_window, p=1, q=1, vol='Garch', dist='Normal', rescale=False)
                last_res = model.fit(disp='off', show_warning=False)
            except Exception as e:
                logging.warning(f"GARCH fit failed at index {i}: {str(e)}. Using last valid model.")
                last_res = None
        
        if last_res and last_res.convergence_flag == 0:
            forecast = last_res.forecast(horizon=1, reindex=False)
            garch_forecasts.loc[current_timestamp] = np.sqrt(forecast.variance.iloc[-1, 0])

    df['garch_vol_forecast'] = garch_forecasts / 100
    df['garch_std_residuals'] = (df['returns'] / df['garch_vol_forecast']).shift(1)
    
    return df

@timer_decorator
def create_enhanced_original_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Creates a refined version of the baseline statistical features."""
    logging.info("Engineering enhanced original feature set...")
    
    df['price_acceleration'] = df['returns'].diff()

    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)

    df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    df['overlap_session'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)

    df['vol_momentum'] = df['garch_vol_forecast'].pct_change(5)

    rolling_max = df['high'].rolling(config.VOL_ROLLING_WINDOW).max()
    rolling_min = df['low'].rolling(config.VOL_ROLLING_WINDOW).min()
    df['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)
    
    df[f'sma_{config.LONG_TERM_WINDOW}'] = df['close'].rolling(window=config.LONG_TERM_WINDOW).mean()
    df[f'price_vs_sma_{config.LONG_TERM_WINDOW}'] = df['close'] / df[f'sma_{config.LONG_TERM_WINDOW}']
    df['body_size'] = np.abs(df['close'] - df['open'])
    df['body_to_range_ratio'] = df['body_size'] / (df['high'] - df['low'])
    
    return df

@numba.jit(nopython=True, cache=True)
def calculate_fractal_dimension_numba(prices: np.ndarray) -> float:
    """Numba-accelerated calculation of Higuchi's Fractal Dimension for a single window."""
    n = len(prices)
    k_max = 10 
    lk = np.zeros(k_max)
    x = np.log(1.0 / np.arange(1, k_max + 1))

    for k in range(1, k_max + 1):
        lm = 0.0
        for m in range(k):
            n_max = int(np.floor((n - m - 1) / k))
            if n_max <= 0: continue
            
            indices = np.arange(n_max + 1) * k + m
            lm += np.sum(np.abs(prices[indices[1:]] - prices[indices[:-1]])) * (n - 1) / (n_max * k)
        lk[k - 1] = np.log(lm / k)
    sum_x = np.sum(x)
    sum_y = np.sum(lk)
    sum_xy = np.sum(x * lk)
    sum_x2 = np.sum(x * x)
    
    slope = (k_max * sum_xy - sum_x * sum_y) / (k_max * sum_x2 - sum_x * sum_x)
    return slope
@timer_decorator
def create_advanced_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Creates a curated set of advanced, non-lagging market features."""
    logging.info("Engineering advanced feature set (Fractal & Microstructure)...")

    df['fractal_dimension'] = df['close'].rolling(
        window=config.FRACTAL_DIMENSION_WINDOW
    ).apply(calculate_fractal_dimension_numba, raw=True)

    candle_range = df['high'] - df['low']
    df['buying_pressure'] = (df['close'] - df['low']) / candle_range
    df['selling_pressure'] = (df['high'] - df['close']) / candle_range
    df['buying_pressure'].fillna(0.5, inplace=True)
    df['selling_pressure'].fillna(0.5, inplace=True)

    return df

@timer_decorator
def create_focused_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers non-lagging context features from the 15M timeframe ONLY."""
    logging.info("Engineering focused multi-timeframe (15M) features...")
    
    agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    df_15m = df.resample('15T').agg(agg_rules).dropna()
    candle_range_15m = df_15m['high'] - df_15m['low']
    candle_body_15m = abs(df_15m['close'] - df_15m['open'])
    
    df_15m['conviction_15m'] = candle_body_15m / candle_range_15m
    df_15m['close_pos_15m'] = (df_15m['close'] - df_15m['low']) / candle_range_15m
    df_15m['return_15m'] = df_15m['close'].pct_change()
    df_15m['conviction_15m'].fillna(0, inplace=True)
    df_15m['close_pos_15m'].fillna(0.5, inplace=True)
    
    features_15m = df_15m[['conviction_15m', 'close_pos_15m', 'return_15m']]
    
    df = df.join(features_15m)
    df[features_15m.columns] = df[features_15m.columns].fillna(method='ffill')
    
    return df

@timer_decorator
def create_focused_sax_feature(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Generates a single, powerful categorical SAX feature for M5 price motifs."""
    logging.info("Engineering focused SAX feature (single categorical)...")
    
    alphabet_cuts = cuts_for_asize(config.SAX_ALPHABET_SIZE)
    sax_words = []
    for i in range(config.SAX_WINDOW_SIZE - 1, len(df)):
        snippet = df['close'].iloc[i - config.SAX_WINDOW_SIZE + 1:i + 1].values
        try:
            sax_word = ts_to_string(znorm(snippet), alphabet_cuts)
            sax_words.append(sax_word)
        except:
            sax_words.append(None)
    sax_series = pd.Series([None] * (config.SAX_WINDOW_SIZE - 1) + sax_words, 
                          index=df.index, name='sax_motif')
    
    df['sax_motif'] = sax_series.astype('category')
    
    return df

@numba.jit(nopython=True, cache=True)
def calculate_target_labels_numba(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volatility: np.ndarray, pt_mult: float, sl_mult: float, time_limit: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    target = np.full(n, np.nan)
    time_to_hit = np.full(n, np.nan)
    
    for i in range(n - time_limit - 1):
        if np.isnan(volatility[i]) or volatility[i] <= 1e-9:
            continue
        
        entry_price = close[i]
        upper_barrier = entry_price * (1 + (volatility[i] * pt_mult))
        lower_barrier = entry_price * (1 - (volatility[i] * sl_mult))
        target[i] = 0 
        time_to_hit[i] = time_limit
        
        for j in range(1, time_limit + 1):
            future_high = high[i + j]
            future_low = low[i + j]
            
            hit_upper = future_high >= upper_barrier
            hit_lower = future_low <= lower_barrier
            
            if hit_upper and hit_lower:
                target[i] = 0
                time_to_hit[i] = j
                break 
            elif hit_upper:
                target[i] = 1 
                time_to_hit[i] = j
                break
            elif hit_lower:
                target[i] = -1
                time_to_hit[i] = j
                break
    return target, time_to_hit

@timer_decorator
def generate_labels(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    logging.info("Defining target variable with Triple-Barrier Method...")
    
    vol_for_labeling = df['garch_vol_forecast'].copy().dropna()
    aligned_data = df.loc[vol_for_labeling.index, ['close', 'high', 'low']]
    aligned_data['volatility'] = vol_for_labeling
    
    targets, time_hits = calculate_target_labels_numba(
        close=aligned_data['close'].values,
        high=aligned_data['high'].values,
        low=aligned_data['low'].values,
        volatility=aligned_data['volatility'].values,
        pt_mult=config.PT_MULT,
        sl_mult=config.SL_MULT,
        time_limit=config.TIME_LIMIT
    )
    
    df['target'] = pd.Series(targets, index=aligned_data.index)
    df['time_to_hit'] = pd.Series(time_hits, index=aligned_data.index)
    
    return df

@timer_decorator
def create_all_v4_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Master orchestrator for the V4 feature engineering pipeline."""
    logging.info("--- Starting Master Feature Engineering Pipeline (V4) ---")
    
    df['returns'] = df['close'].pct_change()
    df = calculate_rolling_garch_optimized(df, config)
    df = create_enhanced_original_features(df, config)
    df = create_advanced_features(df, config)
    df = create_focused_multi_timeframe_features(df)
    df = create_focused_sax_feature(df, config)
    
    return df

def main():
    pipeline_start_time = time.time()
    cfg = Config()
    
    logging.info(f"Loading raw data from: {cfg.RAW_DATA_FILEPATH}")
    try:
        df_raw = pd.read_parquet(cfg.RAW_DATA_FILEPATH)
        logging.info(f"Loaded {len(df_raw):,} raw M5 records.")
    except FileNotFoundError:
        logging.error(f"FATAL: Raw data file not found at {cfg.RAW_DATA_FILEPATH}")
        sys.exit(1)
    df_features = create_all_v4_features(df_raw, cfg)
    df_final = generate_labels(df_features, cfg)

    logging.info("Cleaning and saving final dataset...")
    cols_to_exclude = ['open', 'high', 'low', 'close', 'returns', 'target', 'time_to_hit']
    feature_columns = [col for col in df_final.columns if col not in cols_to_exclude]

    final_ml_df = df_final[feature_columns + ['target', 'time_to_hit']].copy()
    initial_rows = len(final_ml_df)
    final_ml_df.dropna(subset=['target'], inplace=True)
 
    final_ml_df.dropna(subset=feature_columns, inplace=True)
    rows_dropped = initial_rows - len(final_ml_df)
    logging.info(f"Dropped {rows_dropped:,} rows ({rows_dropped/initial_rows:.1%}) due to missing data.")
    final_ml_df['target'] = final_ml_df['target'].astype(int)

    logging.info(f"Final ML-Ready Dataset (V4) Shape: {final_ml_df.shape}")
    logging.info("Final Target Distribution:")
    print(final_ml_df['target'].value_counts(normalize=True).sort_index().to_string())
    cfg.OUTPUT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    final_ml_df.to_parquet(cfg.OUTPUT_FILEPATH, engine='pyarrow', compression='snappy')
    logging.info(f"Successfully saved V4 dataset to: {cfg.OUTPUT_FILEPATH}")

    pipeline_end_time = time.time()
    logging.info(f"--- V4 Pipeline Complete in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
