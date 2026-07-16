import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ("15m", "interval_15m"),
    ("1h", "interval"),
    ("4h", "interval"),
    ("8h", "interval"),
]


def plot_timeseries(period, time_col):
    df = pd.read_csv(RESULTS_DIR / f"predictions_{period}.csv")
    df[time_col] = pd.to_datetime(df[time_col])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(df[time_col], df["volume_true"], label="Gercek", marker="o", ms=3)
    ax1.plot(df[time_col], df["volume_pred_xgb"], label="XGBoost", marker="x", ms=3)
    ax1.set_ylabel("Cagri hacmi")
    ax1.set_title(f"{period} - Volume: Gercek vs Tahmin (17-23 Temmuz 2023)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df[time_col], df["AHT_true"], label="Gercek", marker="o", ms=3)
    ax2.plot(df[time_col], df["AHT_pred_xgb"], label="XGBoost", marker="x", ms=3)
    ax2.set_ylabel("AHT (saniye)")
    ax2.set_xlabel("Zaman")
    ax2.set_title(f"{period} - AHT: Gercek vs Tahmin")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    out = PLOTS_DIR / f"timeseries_{period}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("Kaydedildi:", out)


def plot_scatter(period, time_col):
    df = pd.read_csv(RESULTS_DIR / f"predictions_{period}.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, target, unit in [
        (ax1, "volume", "cagri"),
        (ax2, "AHT", "saniye"),
    ]:
        true = df[f"{target}_true"]
        pred = df[f"{target}_pred_xgb"]
        ax.scatter(true, pred, alpha=0.6, s=20)
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", label="y = x")
        ax.set_xlabel(f"Gercek {target} ({unit})")
        ax.set_ylabel(f"Tahmin {target} ({unit})")
        ax.set_title(f"{period} - {target}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"scatter_{period}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("Kaydedildi:", out)


def plot_mae_comparison():
    m = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")

    for target in ["volume", "AHT"]:
        sub = m[m["target"] == target].reset_index(drop=True)
        x = range(len(sub))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar([i - width for i in x], sub["naive_mae"], width, label="Naive")
        plt.bar(list(x), sub["seasonal_mae"], width, label="Seasonal")
        plt.bar([i + width for i in x], sub["xgboost_mae"], width, label="XGBoost")
        plt.xticks(list(x), sub["period"])
        plt.ylabel("MAE")
        plt.title(f"{target} MAE - Model Karsilastirmasi")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        out = PLOTS_DIR / f"mae_{target}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print("Kaydedildi:", out)


if __name__ == "__main__":
    for period, time_col in PERIODS:
        plot_timeseries(period, time_col)
        plot_scatter(period, time_col)
    plot_mae_comparison()
    print("\nTum grafikler kaydedildi:", PLOTS_DIR)
