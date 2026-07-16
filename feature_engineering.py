import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PERIODS = [
    ("15m", "series_15m_9to5.csv", "interval_15m", 32, [8, 16, 32]),
    ("1h", "series_1h_9to5.csv", "interval", 8, [4, 8]),
    ("4h", "series_4h_9to5.csv", "interval", 2, [2, 4]),
    ("8h", "series_8h_9to5.csv", "interval", 1, [2, 4]),
]


def add_features(df, time_col, periods_per_day, aht_median_windows=None):
    df = df.sort_values(time_col).reset_index(drop=True)

    ts = pd.to_datetime(df[time_col])
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Dongusel (sin-cos) kodlama: saat / gun / ay sinirlari dogru temsil edilsin
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df["volume_lag1"] = df["volume"].shift(1)
    df["volume_lag2"] = df["volume"].shift(2)
    df["AHT_lag1"] = df["AHT"].shift(1)
    df["AHT_lag2"] = df["AHT"].shift(2)

    df["volume_lag_day"] = df["volume"].shift(periods_per_day)
    df["AHT_lag_day"] = df["AHT"].shift(periods_per_day)

    df["volume_lag_week"] = df["volume"].shift(periods_per_day * 7)
    df["AHT_lag_week"] = df["AHT"].shift(periods_per_day * 7)

    df["volume_roll_mean_4"] = df["volume"].shift(1).rolling(4).mean()
    df["AHT_roll_mean_4"] = df["AHT"].shift(1).rolling(4).mean()


    df["AHT_roll_mean_8"] = df["AHT"].shift(1).rolling(8).mean()
    df["AHT_roll_std_4"] = df["AHT"].shift(1).rolling(4).std()
    df["volume_roll_std_4"] = df["volume"].shift(1).rolling(4).std()

    shifted_aht = df["AHT"].shift(1)
    for window in aht_median_windows or [8, 16]:
        df[f"AHT_roll_median_{window}"] = shifted_aht.rolling(window).median()

    return df


for name, filename, time_col, periods_per_day, median_windows in PERIODS:
    input_path = DATA_DIR / filename
    output_path = DATA_DIR / f"features_{name}_9to5.csv"

    df = pd.read_csv(input_path)
    df = add_features(df, time_col, periods_per_day, median_windows)

    before = len(df)
    df = df.dropna().reset_index(drop=True)

    df.to_csv(output_path, index=False)

    print(f"\n{name} feature dosyası kaydedildi: {output_path}")
    print(f"{name} satır (once): {before} -> (sonra): {len(df)}")
    print(df.head(3))
