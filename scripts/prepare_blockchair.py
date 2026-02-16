import pandas as pd

df = pd.read_csv(
    "data/raw/transactions_20251216.csv",
    sep="\t",
    engine="python"
)

# Select features used in the paper
selected_cols = [
    "hash",
    "time",
    "input_count",
    "output_count",
    "input_total",
    "output_total",
    "fee",
    "fee_per_kb",
    "fee_per_kwu",
    "cdd_total"
]

df = df[selected_cols]

# Convert time
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])

# Convert numeric columns safely
num_cols = [
    "input_count", "output_count",
    "input_total", "output_total",
    "fee", "fee_per_kb", "fee_per_kwu",
    "cdd_total"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

df.to_csv("data/processed/blockchair_clean.csv", index=False)

print("blockchair_clean.csv saved")
print("Shape:", df.shape)
