import numpy as np
import pandas as pd
from pathlib import Path

input_path = Path(r"C:\Users\mu_sa\PycharmProjects\callcenter-aht-forecast\data\calls_clean.csv")
output_dir = Path(r"C:\Users\mu_sa\PycharmProjects\callcenter-aht-forecast\data")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)
df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
df["talk_time"] = pd.to_numeric(df["talk_time"], errors="coerce")
df = df[df["time_stamp"].notna()]

start_t = pd.to_datetime("09:00").time()
end_t = pd.to_datetime("17:00").time()
df = df[df["time_stamp"].dt.time.between(start_t, end_t, inclusive="left")]

df["interval_15m"] = df["time_stamp"].dt.floor("15min")

vol = df.groupby("interval_15m").size().rename("volume")
answered = df[(df["Answered"] == "Y") & (df["talk_time"] > 0)]
aht = answered.groupby("interval_15m")["talk_time"].mean().rename("AHT")

agg = pd.concat([vol, aht], axis=1).reset_index()

min_day = agg["interval_15m"].min().normalize()
max_day = agg["interval_15m"].max().normalize()
days = pd.date_range(min_day, max_day, freq="D")

grid_list = []
for d in days:
    grid_list.append(
        pd.date_range(
            d + pd.Timedelta(hours=9),
            d + pd.Timedelta(hours=16, minutes=45),
            freq="15min",
        )
    )

grid = pd.DataFrame({"interval_15m": np.concatenate(grid_list)})
series_15m = grid.merge(agg, on="interval_15m", how="left")
series_15m["volume"] = series_15m["volume"].fillna(0).astype(int)

def weighted_aht(group):
    total_vol = group["volume"].sum()
    if total_vol == 0:
        return np.nan
    valid = group[group["AHT"].notna() & (group["volume"] > 0)]
    if valid.empty:
        return np.nan
    return (valid["AHT"] * valid["volume"]).sum() / valid["volume"].sum()

def make_period(df_in, freq_name, floor_rule):
    tmp = df_in.copy()
    tmp["interval"] = tmp["interval_15m"].dt.floor(floor_rule)

    out = tmp.groupby("interval").apply(
        lambda g: pd.Series(
            {
                "volume": g["volume"].sum(),
                "AHT": weighted_aht(g),
            }
        )
    ).reset_index()

    out_path = output_dir / f"series_{freq_name}_9to5.csv"
    out.to_csv(out_path, index=False)
    print(f"{freq_name} kaydedildi:", out_path)
    print(f"{freq_name} satir sayisi:", len(out))
    return out

path_15m = output_dir / "series_15m_9to5.csv"
series_15m.to_csv(path_15m, index=False)
print("15m kaydedildi:", path_15m)
print("15m satir sayisi:", len(series_15m))

daily_15m = series_15m.groupby(series_15m["interval_15m"].dt.date).size()
print("\n15m gunluk satir dagilimi:")
print(daily_15m.value_counts().sort_index())

series_1h = make_period(series_15m, "1h", "h")
series_6h = make_period(series_15m, "6h", "6h")

print("\nOrnek kayitlar:")
print(series_15m.head(5))
print(series_1h.head(5))
print(series_6h.head(5))