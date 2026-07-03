import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

file_path = DATA_DIR / "callCenter.data.csv"
output_path = DATA_DIR / "calls_clean.csv"

df = pd.read_csv(file_path)
print(df.shape)
print(df.columns.tolist())

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["time_stamp"] = pd.to_datetime(
    df["time_stamp"],
    format="%m/%d/%y %I:%M %p",
    errors="coerce"
)
df["talk_time"] = pd.to_numeric(df["talk_time"], errors="coerce")

print("Toplam satır:", len(df))
print("Parse edilemeyen time_stamp:", df["time_stamp"].isna().sum())
print("NaN talk_time:", df["talk_time"].isna().sum())
print("Negatif talk_time:", (df["talk_time"] < 0).sum())
print("\nAnswered dağılımı:")
print(df["Answered"].value_counts(dropna=False))

df = df.drop(columns=["first_name", "last_name", "calling_number"])

before_rows = len(df)
df = df[df["time_stamp"].notna()]
df = df[df["talk_time"].notna()]
df = df[df["talk_time"] >= 0]
df = df[df["Answered"].isin(["Y", "N"])]

print("\nTemizlik sonrası satır:", len(df))
print("Silinen satır:", before_rows - len(df))
print("Tarih aralığı:", df["time_stamp"].min(), "->", df["time_stamp"].max())

aht_valid = df[(df["Answered"] == "Y") & (df["talk_time"] > 0)]
print("AHT için uygun çağrı:", len(aht_valid))
print("Ortalama talk_time:", round(aht_valid["talk_time"].mean(), 2))

df.to_csv(output_path, index=False)
print("Kaydedildi:", output_path)