import pandas as pd
from pathlib import Path

# Girdi / çıktı
input_path = Path(r"C:\Users\mu_sa\PycharmProjects\callcenter-aht-forecast\data\calls_clean.csv")
output_path = Path(r"C:\Users\mu_sa\PycharmProjects\callcenter-aht-forecast\data\series_15m_9to5.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Veri
df = pd.read_csv(input_path)
df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
df["talk_time"] = pd.to_numeric(df["talk_time"], errors="coerce")

# 09:00–17:00 (17:00 dahil değil)
start_t = pd.to_datetime("09:00").time()
end_t = pd.to_datetime("17:00").time()
df = df[df["time_stamp"].notna()]
df = df[df["time_stamp"].dt.time.between(start_t, end_t, inclusive="left")]

# 15 dk interval
df["interval_15m"] = df["time_stamp"].dt.floor("15min")

# Volume: tüm çağrılar (Y+N)
vol = df.groupby("interval_15m").size().rename("volume_15m")

# AHT: sadece cevaplanan + talk_time>0
aht_src = df[(df["Answered"] == "Y") & (df["talk_time"] > 0)]
aht = aht_src.groupby("interval_15m")["talk_time"].mean().rename("AHT_15m")

series = pd.concat([vol, aht], axis=1).reset_index()

# ---- Sabit grid: her gün 09:00-16:45 arası 32 satır ----
min_day = series["interval_15m"].dt.normalize().min()
max_day = series["interval_15m"].dt.normalize().max()
days = pd.date_range(min_day, max_day, freq="D")

all_intervals = []
for d in days:
    intr = pd.date_range(
        d + pd.Timedelta(hours=9),
        d + pd.Timedelta(hours=16, minutes=45),
        freq="15min"
    )
    all_intervals.append(intr)

grid = pd.DataFrame({"interval_15m": pd.DatetimeIndex([]).append(all_intervals)})
series = grid.merge(series, on="interval_15m", how="left")

# Boş periyotlar
series["volume_15m"] = series["volume_15m"].fillna(0).astype(int)
# AHT boş kalabilir (volume=0 veya answered yoksa NaN)

# Yardımcı kolonlar
series["date"] = series["interval_15m"].dt.date
series["time"] = series["interval_15m"].dt.time

# Kontrol: her gün 32 satır
daily_rows = series.groupby("date").size()
print("Günlük satır dağılımı:")
print(daily_rows.value_counts().sort_index())  # ideal: sadece 32 -> N gün

print("Toplam satır:", len(series))
print("Tarih aralığı:", series["interval_15m"].min(), "->", series["interval_15m"].max())
print(series.head(10))

series.to_csv(output_path, index=False)
print("Kaydedildi:", output_path)