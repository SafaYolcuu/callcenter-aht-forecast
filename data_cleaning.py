import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_path = Path(r"C:\Users\mu_sa\OneDrive\Masaüstü\CallCenterDataSets\Time Series\Synthetic Call Center\raw_data\callCenter.data.csv")

df = pd.read_csv(input_path)
df["time_stamp"] = pd.to_datetime(df["time_stamp"])
df["date"] = df["time_stamp"].dt.date

daily_counts = df.groupby("date").size().reset_index(name="row_count")

print("Toplam gün sayısı:", daily_counts.shape[0])
print("Günlük ortalama satır:", round(daily_counts["row_count"].mean(), 2))
print("En az satır:", daily_counts["row_count"].min())
print("En çok satır:", daily_counts["row_count"].max())

print("\nİlk 10 gün:")
print(daily_counts.head(10))

print("\nÖzet istatistik:")
print(daily_counts["row_count"].describe())

# mesai filtresi
mesai_start = pd.to_datetime("08:00").time()
mesai_end = pd.to_datetime("21:30").time()

df_mesai = df[df["time_stamp"].dt.time.between(mesai_start, mesai_end)]
daily_mesai = df_mesai.groupby(df_mesai["time_stamp"].dt.date).size().reset_index(name="row_count")

print("\nMesai içi günlük ortalama:", round(daily_mesai["row_count"].mean(), 2))
print("Mesai içi en az:", daily_mesai["row_count"].min())
print("Mesai içi en çok:", daily_mesai["row_count"].max())

# grafik
plt.figure(figsize=(12, 4))
plt.plot(daily_counts["date"], daily_counts["row_count"])
plt.title("Günlük çağrı sayısı")
plt.xlabel("Tarih")
plt.ylabel("Satır sayısı")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()