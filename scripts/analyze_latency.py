import pandas as pd

df = pd.read_csv("data/latency_raw.csv")
columns = ["first_token_ms", "total_ms", "tokens", "tokens_per_sec"]

print("Average per model:")
print(df.groupby("model")[columns].mean().round(1))

print("\nMedian total time (ms):")
print(df.groupby("model")["total_ms"].median())