import itertools
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TEST_START_DATE = "2023-07-17"
TEST_END_DATE_EXCLUSIVE = "2023-07-24"
VAL_DAYS = 14  #

PERIODS = [
    ("15m", "features_15m_9to5.csv", "interval_15m"),
    ("1h", "features_1h_9to5.csv", "interval"),
    ("4h", "features_4h_9to5.csv", "interval"),
    ("8h", "features_8h_9to5.csv", "interval"),
]


VOLUME_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [5, 7],
    "learning_rate": [0.03, 0.08],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "min_child_weight": [3],
    "reg_lambda": [1.0, 10.0],
}


VOLUME_PARAM_GRIDS = {
    "15m": {
        "n_estimators": [500, 800],
        "max_depth": [5, 7],
        "learning_rate": [0.05, 0.08],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "min_child_weight": [1, 3],
        "reg_lambda": [1.0, 5.0],
    },
}

AHT_PARAM_GRIDS = {
    "15m": {
        "n_estimators": [600, 800],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.02],
        "subsample": [0.9, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "min_child_weight": [1, 3],
        "reg_lambda": [1.0, 5.0],
        "reg_alpha": [0.0, 0.5],
        "objective": ["reg:squarederror", "reg:absoluteerror"],
    },
    "1h": {
        "n_estimators": [600, 800],
        "max_depth": [6, 8],
        "learning_rate": [0.01, 0.02],
        "subsample": [0.9, 1.0],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 2],
        "reg_lambda": [1.0, 5.0],
        "reg_alpha": [0.0, 1.0],
    },
    "4h": {
        "n_estimators": [800, 1000],
        "max_depth": [8, 10],
        "learning_rate": [0.01, 0.02],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
        "min_child_weight": [1],
        "reg_lambda": [1.0, 5.0],
        "reg_alpha": [0.0, 0.5, 1.0],
    },
    "8h": {
        "n_estimators": [600, 800],
        "max_depth": [7, 8],
        "learning_rate": [0.02, 0.03],
        "subsample": [1.0],
        "colsample_bytree": [0.9],
        "min_child_weight": [1],
        "reg_lambda": [0.5, 5.0],
        "reg_alpha": [0.0, 0.5],
    },
}

PARAM_KEYS = [
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha", "objective",
]

SELECTED_PERIODS = set(sys.argv[1:]) if len(sys.argv) > 1 else None
PERIODS_TO_RUN = [p for p in PERIODS if SELECTED_PERIODS is None or p[0] in SELECTED_PERIODS]


def load_cached_volume_params():
    path = RESULTS_DIR / "best_params.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    vol = df[df["target"] == "volume"]
    cached = {}
    for _, row in vol.iterrows():
        params = {k: row[k] for k in PARAM_KEYS if k in row and pd.notna(row[k])}
        cached[row["period"]] = params
    return cached


CACHED_VOLUME_PARAMS = load_cached_volume_params()

RANDOM_SEED = 42


def all_param_combinations(grid):
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def safe_to_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"Uyari: {path.name} kilitli, bunun yerine yazildi: {fallback.name}")
        return fallback


def tune_xgb_grid_search(X_fit, y_fit, X_val, y_val, param_grid):
    best_params = None
    best_mae = float("inf")
    combinations = all_param_combinations(param_grid)

    for i, params in enumerate(combinations, start=1):
        model = XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        if mae < best_mae:
            best_mae = mae
            best_params = params
        if i % 20 == 0 or i == len(combinations):
            print(f"  grid ilerleme: {i}/{len(combinations)} | en iyi val MAE: {best_mae:.4f}")
            sys.stdout.flush()

    sys.stdout.flush()
    return best_params, best_mae, len(combinations)


def merge_period_results(df_new, path, period_names):
    if not path.exists() or not period_names:
        return df_new
    df_old = pd.read_csv(path)
    df_old = df_old[~df_old["period"].isin(period_names)]
    return pd.concat([df_old, df_new], ignore_index=True)


all_metrics = []
best_params_rows = []

for period_name, filename, time_col in PERIODS_TO_RUN:
    input_path = DATA_DIR / filename
    df = pd.read_csv(input_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    test_start_ts = pd.Timestamp(TEST_START_DATE)
    test_end_ts = pd.Timestamp(TEST_END_DATE_EXCLUSIVE)
    val_start_ts = test_start_ts - pd.Timedelta(days=VAL_DAYS)


    df = df[df[time_col] < test_end_ts].copy()

    train_df = df[df[time_col] < test_start_ts].copy()
    test_df = df[(df[time_col] >= test_start_ts) & (df[time_col] < test_end_ts)].copy()


    train_fit_df = train_df[train_df[time_col] < val_start_ts].copy()
    val_df = train_df[train_df[time_col] >= val_start_ts].copy()

    feature_cols = [
        c for c in df.columns
        if c not in [time_col, "volume", "AHT"]
    ]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    X_fit = train_fit_df[feature_cols]
    X_val = val_df[feature_cols]

    y_train_volume = train_df["volume"]
    y_test_volume = test_df["volume"]
    y_fit_volume = train_fit_df["volume"]
    y_val_volume = val_df["volume"]

    y_train_aht = train_df["AHT"]
    y_test_aht = test_df["AHT"]
    y_fit_aht = train_fit_df["AHT"]
    y_val_aht = val_df["AHT"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_tune = StandardScaler()
    X_fit_scaled = scaler_tune.fit_transform(X_fit)
    X_val_scaled = scaler_tune.transform(X_val)


    naive_volume_pred = test_df["volume_lag1"]
    seasonal_volume_pred = test_df["volume_lag_week"]
    naive_aht_pred = test_df["AHT_lag1"]
    seasonal_aht_pred = test_df["AHT_lag_week"]

    naive_volume_mae = mean_absolute_error(y_test_volume, naive_volume_pred)
    seasonal_volume_mae = mean_absolute_error(y_test_volume, seasonal_volume_pred)
    naive_aht_mae = mean_absolute_error(y_test_aht, naive_aht_pred)
    seasonal_aht_mae = mean_absolute_error(y_test_aht, seasonal_aht_pred)

    print(f"\n=== {period_name} hiperparametre aramasi ===")
    print(f"fit: {len(train_fit_df)} | val: {len(val_df)} | test: {len(test_df)}")

 
    vol_grid = VOLUME_PARAM_GRIDS.get(period_name, VOLUME_PARAM_GRID)
    use_cached_volume = (
        period_name in CACHED_VOLUME_PARAMS
        and period_name not in VOLUME_PARAM_GRIDS
    )

    if use_cached_volume:
        vol_params = CACHED_VOLUME_PARAMS[period_name]
        model_vol_val = XGBRegressor(**vol_params, random_state=RANDOM_SEED, n_jobs=-1)
        model_vol_val.fit(X_fit_scaled, y_fit_volume)
        vol_val_mae = mean_absolute_error(y_val_volume, model_vol_val.predict(X_val_scaled))
        print("volume onceki en iyi parametreler kullanildi (grid atlandi)")
    else:
        vol_params, vol_val_mae, vol_grid_size = tune_xgb_grid_search(
            X_fit_scaled, y_fit_volume, X_val_scaled, y_val_volume, vol_grid
        )
        print(f"volume grid denenen kombinasyon: {vol_grid_size}")
    print(f"volume best val MAE: {vol_val_mae:.4f} | params: {vol_params}")

    # AHT: tek validation penceresi ile periyot bazli grid search
    aht_grid = AHT_PARAM_GRIDS[period_name]
    aht_params, aht_val_mae, aht_grid_size = tune_xgb_grid_search(
        X_fit_scaled, y_fit_aht, X_val_scaled, y_val_aht, aht_grid
    )
    print(f"AHT    grid denenen kombinasyon: {aht_grid_size}")
    print(f"AHT    best val MAE: {aht_val_mae:.4f} | params: {aht_params}")

    best_params_rows.append({"period": period_name, "target": "volume", "val_mae": vol_val_mae, **vol_params})
    best_params_rows.append({"period": period_name, "target": "AHT", "val_mae": aht_val_mae, **aht_params})

    # Secilen parametrelerle tum train uzerinde final model
    model_volume = XGBRegressor(**vol_params, random_state=RANDOM_SEED, n_jobs=-1)
    model_volume.fit(X_train_scaled, y_train_volume)
    xgb_volume_pred = model_volume.predict(X_test_scaled)
    xgb_volume_mae = mean_absolute_error(y_test_volume, xgb_volume_pred)

    model_aht = XGBRegressor(**aht_params, random_state=RANDOM_SEED, n_jobs=-1)
    model_aht.fit(X_train_scaled, y_train_aht)
    xgb_aht_pred = model_aht.predict(X_test_scaled)
    xgb_aht_mae = mean_absolute_error(y_test_aht, xgb_aht_pred)

    all_metrics.extend([
        {
            "period": period_name,
            "target": "volume",
            "naive_mae": naive_volume_mae,
            "seasonal_mae": seasonal_volume_mae,
            "xgboost_mae": xgb_volume_mae,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        {
            "period": period_name,
            "target": "AHT",
            "naive_mae": naive_aht_mae,
            "seasonal_mae": seasonal_aht_mae,
            "xgboost_mae": xgb_aht_mae,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
    ])

    pred_out = pd.DataFrame({
        time_col: test_df[time_col],
        "volume_true": y_test_volume,
        "volume_pred_xgb": xgb_volume_pred,
        "volume_pred_naive": naive_volume_pred,
        "volume_pred_seasonal": seasonal_volume_pred,
        "AHT_true": y_test_aht,
        "AHT_pred_xgb": xgb_aht_pred,
        "AHT_pred_naive": naive_aht_pred,
        "AHT_pred_seasonal": seasonal_aht_pred,
    })
    pred_path = RESULTS_DIR / f"predictions_{period_name}.csv"
    saved_pred_path = safe_to_csv(pred_out, pred_path)

    print(f"{period_name} test sonuclari")
    print(f"volume  -> basit: {naive_volume_mae:.4f} | mevsimsel: {seasonal_volume_mae:.4f} | xgb: {xgb_volume_mae:.4f}")
    print(f"AHT     -> basit: {naive_aht_mae:.4f} | mevsimsel: {seasonal_aht_mae:.4f} | xgb: {xgb_aht_mae:.4f}")
    print(f"tahmin dosyasi: {saved_pred_path}")

metrics_df = pd.DataFrame(all_metrics)
metrics_path = RESULTS_DIR / "metrics_summary.csv"
saved_metrics_path = safe_to_csv(metrics_df, metrics_path)

params_df = pd.DataFrame(best_params_rows)
params_path = RESULTS_DIR / "best_params.csv"
saved_params_path = safe_to_csv(params_df, params_path)

print("\nOzet metrik tablosu:")
print(metrics_df)
print(f"\nKaydedildi: {saved_metrics_path}")
print(f"En iyi parametreler: {saved_params_path}")
