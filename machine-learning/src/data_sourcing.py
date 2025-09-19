import argparse
import logging
import sys
import zipfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
M1_COLUMNS = ["datetime_str", "open", "high", "low", "close", "volume"]
TICK_COLUMNS = ["datetime_str", "bid", "ask", "volume"]


def process_histdata_zips(input_dir: Path, output_dir: Path) -> None:
    logging.info(f"Starting Histdata processing from input directory: {input_dir}")

    zip_files = sorted(list(input_dir.glob("*.zip")))
    if not zip_files:
        logging.error(f"No .zip files found in directory: {input_dir}")
        logging.error("Please download data from Histdata.com and place it in the input directory.")
        sys.exit(1)

    logging.info(f"Found {len(zip_files)} ZIP files to process.")

    all_months_df_list = []
    is_tick_data = None 

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_filename = [name for name in zf.namelist() if name.endswith(".csv")][0]
                logging.info(f"Processing: {zip_path.name} -> {csv_filename}")
                if is_tick_data is None: 
                    if "_T_" in csv_filename:
                        is_tick_data = True
                        logging.info("Detected Tick Data format.")
                    else:
                        is_tick_data = False
                        logging.info("Detected M1 Bar Data format.")
                if is_tick_data:
                    delimiter = ","
                    columns = TICK_COLUMNS
                else:
                    delimiter = ";"
                    columns = M1_COLUMNS

                csv_content = zf.read(csv_filename).decode("utf-8")
                csv_buffer = StringIO(csv_content)

                month_df = pd.read_csv(
                    csv_buffer,
                    delimiter=delimiter,
                    header=None,
                    names=columns,
                    on_bad_lines="warn",
                )
                all_months_df_list.append(month_df)

        except Exception as e:
            logging.error(f"Failed to process file {zip_path}: {e}")
            continue

    if not all_months_df_list:
        logging.error("No data could be extracted from the provided ZIP files.")
        sys.exit(1)
    logging.info("Combining monthly dataframes and processing timestamps...")
    df = pd.concat(all_months_df_list, ignore_index=True)
    datetime_format = "%Y%m%d %H%M%S%f" if is_tick_data else "%Y%m%d %H%M%S"
    df["datetime"] = pd.to_datetime(df["datetime_str"], format=datetime_format)

    df["datetime"] = df["datetime"].dt.tz_localize("Etc/GMT+5")
    df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    df.set_index("datetime", inplace=True)
    df.drop(columns=["datetime_str"], inplace=True)
    df.sort_index(inplace=True)

    logging.info(f"Successfully processed {len(df)} initial records.")
    logging.info("Resampling data to M5 timeframe...")
    
    if is_tick_data:
        ohlc_dict = {
            "bid": "ohlc",
            "volume": "sum"
        }
        df_m5 = df.resample("5T", label="right").apply(ohlc_dict)
        df_m5.columns = df_m5.columns.droplevel(0)
    else:
        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_m5 = df.resample("5T", label="right").agg(ohlc_dict)
    df_m5.dropna(how="all", inplace=True)
    logging.info(f"Resampling complete. Generated {len(df_m5)} M5 records.")

    logging.info("Performing gap validation on M5 data...")
    expected_range = pd.date_range(start=df_m5.index.min(), end=df_m5.index.max(), freq='5min')
    missing_timestamps = expected_range.difference(df_m5.index)
    missing_weekdays = [ts for ts in missing_timestamps if ts.weekday() < 5]

    if len(missing_weekdays) > 288:
        logging.warning(f"CRITICAL DATA GAP: Found {len(missing_weekdays)} missing 5-minute bars on weekdays.")
        logging.warning("This indicates a significant data integrity issue that could corrupt the model.")
        logging.warning(f"First 10 missing timestamps: {missing_weekdays[:10]}")
    else:
        logging.info("Data validation: No significant data gaps found on weekdays.")
    nan_count = df_m5.isnull().sum().sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values after resampling. Forward-filling.")
        df_m5.fillna(method='ffill', inplace=True)
    else:
        logging.info("Data validation: No NaN values found in M5 data.")
    try:
        logging.info("Saving validated M5 data to Parquet file...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_date_str = df_m5.index.min().strftime('%Y-%m-%d')
        end_date_str = df_m5.index.max().strftime('%Y-%m-%d')
        
        filename = f"EURUSD_M5_{start_date_str}_{end_date_str}.parquet"
        output_path = output_dir / filename

        df_m5.to_parquet(output_path, engine="pyarrow", compression="snappy")
        logging.info(f"Successfully saved data to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to save data to file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Histdata.com Data Processor for ML Trading Research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="The directory containing downloaded .zip files from Histdata.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory to save the final M5 Parquet file.",
    )

    args = parser.parse_args()

    input_dir_path = Path(args.input_dir)
    output_dir_path = Path(args.output_dir)

    process_histdata_zips(
        input_dir=input_dir_path,
        output_dir=output_dir_path,
    )

if __name__ == "__main__":
    main()
