from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

PERIODS = {
    "15m": "interval_15m",
    "1h": "interval",
    "4h": "interval",
    "8h": "interval",
}

TEST_WINDOW = "17–23 Temmuz 2023"
MODEL_LABELS = {
    "naive": "Naive (lag1)",
    "seasonal": "Seasonal (lag week)",
    "xgboost": "XGBoost",
}


@st.cache_data
def load_metrics():
    path = RESULTS_DIR / "metrics_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ["naive_mae", "seasonal_mae", "xgboost_mae"]:
        df[col] = df[col].round(2)
    return df


@st.cache_data
def load_predictions(period: str):
    path = RESULTS_DIR / f"predictions_{period}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    time_col = PERIODS[period]
    df[time_col] = pd.to_datetime(df[time_col])
    return df, time_col


def metrics_for_period(metrics: pd.DataFrame, period: str) -> pd.DataFrame:
    sub = metrics[metrics["period"] == period].copy()
    rows = []
    for _, row in sub.iterrows():
        target = row["target"]
        unit = "cagri" if target == "volume" else "saniye"
        for model_key, col in [
            ("naive", "naive_mae"),
            ("seasonal", "seasonal_mae"),
            ("xgboost", "xgboost_mae"),
        ]:
            rows.append(
                {
                    "Hedef": target.upper() if target == "AHT" else "Volume",
                    "Model": MODEL_LABELS[model_key],
                    "MAE": row[col],
                    "Birim": unit,
                }
            )
    return pd.DataFrame(rows)


def plot_mae_bars(metrics: pd.DataFrame, target: str):
    sub = metrics[metrics["target"] == target].reset_index(drop=True)
    x = range(len(sub))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], sub["naive_mae"], width, label=MODEL_LABELS["naive"])
    ax.bar(list(x), sub["seasonal_mae"], width, label=MODEL_LABELS["seasonal"])
    ax.bar([i + width for i in x], sub["xgboost_mae"], width, label=MODEL_LABELS["xgboost"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["period"])
    ax.set_ylabel("MAE")
    unit = "cagri" if target == "volume" else "saniye"
    ax.set_title(f"{target.upper()} MAE karsilastirmasi ({unit})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_timeseries(df: pd.DataFrame, time_col: str, period: str, target: str):
    unit = "cagri" if target == "volume" else "saniye"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[time_col], df[f"{target}_true"], label="Gercek", marker="o", ms=3)
    ax.plot(
        df[time_col],
        df[f"{target}_pred_naive"],
        label=MODEL_LABELS["naive"],
        marker=".",
        ms=3,
        alpha=0.8,
    )
    ax.plot(
        df[time_col],
        df[f"{target}_pred_seasonal"],
        label=MODEL_LABELS["seasonal"],
        marker=".",
        ms=3,
        alpha=0.8,
    )
    ax.plot(
        df[time_col],
        df[f"{target}_pred_xgb"],
        label=MODEL_LABELS["xgboost"],
        marker="x",
        ms=4,
    )
    ax.set_ylabel(f"{target} ({unit})")
    ax.set_xlabel("Zaman")
    ax.set_title(f"{period} - {target.upper()}: Gercek vs Tahmin ({TEST_WINDOW})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, period: str, target: str):
    unit = "cagri" if target == "volume" else "saniye"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, model_key in zip(axes, ["naive", "seasonal", "xgboost"]):
        suffix = {"naive": "naive", "seasonal": "seasonal", "xgboost": "xgb"}[model_key]
        true = df[f"{target}_true"]
        pred = df[f"{target}_pred_{suffix}"]
        ax.scatter(true, pred, alpha=0.6, s=20)
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", label="y = x")
        ax.set_xlabel(f"Gercek ({unit})")
        ax.set_ylabel(f"Tahmin ({unit})")
        ax.set_title(MODEL_LABELS[model_key])
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{period} - {target.upper()} scatter")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Call Center Tahmin",
        page_icon="📞",
        layout="wide",
    )

    st.title("Call Center Volume & AHT Tahmin Paneli")
    st.caption(f"Test penceresi: {TEST_WINDOW} | Calisma saati: 09:00–17:00")

    metrics = load_metrics()
    if metrics is None:
        st.error("`results/metrics_summary.csv` bulunamadi. Once `python train_xgboost.py` calistirin.")
        st.stop()

    st.sidebar.header("Ayarlar")
    page = st.sidebar.radio(
        "Sayfa",
        ["Ozet", "Periyot detayi", "Kayitli grafikler"],
    )
    period = st.sidebar.selectbox("Periyot", list(PERIODS.keys()), index=0)

    if page == "Ozet":
        st.subheader("Tum modellerin MAE ozeti")
        display = metrics.copy()
        display["target"] = display["target"].str.upper()
        display = display.rename(
            columns={
                "period": "Periyot",
                "target": "Hedef",
                "naive_mae": "Naive MAE",
                "seasonal_mae": "Seasonal MAE",
                "xgboost_mae": "XGBoost MAE",
                "train_rows": "Train satir",
                "test_rows": "Test satir",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_mae_bars(metrics, "volume"))
        with col2:
            st.pyplot(plot_mae_bars(metrics, "AHT"))

        best = []
        for (period_name, target), grp in metrics.groupby(["period", "target"]):
            row = grp.iloc[0]
            scores = {
                MODEL_LABELS["naive"]: row["naive_mae"],
                MODEL_LABELS["seasonal"]: row["seasonal_mae"],
                MODEL_LABELS["xgboost"]: row["xgboost_mae"],
            }
            winner = min(scores, key=scores.get)
            best.append(
                {
                    "Periyot": period_name,
                    "Hedef": target.upper() if target == "AHT" else "Volume",
                    "En iyi model": winner,
                    "MAE": round(scores[winner], 2),
                }
            )
        st.subheader("En iyi model (MAE)")
        st.dataframe(pd.DataFrame(best), use_container_width=True, hide_index=True)

    elif page == "Periyot detayi":
        st.subheader(f"{period} sonuclari")
        period_metrics = metrics_for_period(metrics, period)
        st.dataframe(period_metrics, use_container_width=True, hide_index=True)

        loaded = load_predictions(period)
        if loaded is None:
            st.warning(f"`results/predictions_{period}.csv` bulunamadi.")
            st.stop()
        df, time_col = loaded

        tab_vol, tab_aht = st.tabs(["Volume", "AHT"])
        with tab_vol:
            st.pyplot(plot_timeseries(df, time_col, period, "volume"))
            st.pyplot(plot_scatter(df, period, "volume"))
        with tab_aht:
            st.pyplot(plot_timeseries(df, time_col, period, "AHT"))
            st.pyplot(plot_scatter(df, period, "AHT"))

        with st.expander("Ham tahmin verisi"):
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.subheader("Kayitli PNG grafikler")
        st.caption("`visualize.py` ile uretilen dosyalar")
        if not PLOTS_DIR.exists():
            st.warning("`results/plots/` klasoru yok. `python visualize.py` calistirin.")
            st.stop()

        png_files = sorted(PLOTS_DIR.glob("*.png"))
        if not png_files:
            st.warning("Grafik dosyasi bulunamadi.")
            st.stop()

        filter_period = st.selectbox("Filtre (periyot)", ["Tumu"] + list(PERIODS.keys()))
        for path in png_files:
            if filter_period != "Tumu" and filter_period not in path.name:
                continue
            st.image(str(path), caption=path.name, use_container_width=True)


if __name__ == "__main__":
    main()
