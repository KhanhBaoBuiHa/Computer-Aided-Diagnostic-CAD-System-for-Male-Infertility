import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

init_db()

conn = sqlite3.connect("predictions.db")
df = pd.read_sql("SELECT * FROM predictions", conn)
conn.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df["binary_label"].value_counts().plot(kind="bar", ax=axes[0], title="Binary Prediction Distribution")
df["binary_conf"].plot(kind="hist", bins=20, ax=axes[1], title="Confidence Distribution")
df["latency_ms"].plot(kind="hist", bins=20, ax=axes[2], title="Latency Distribution (ms)")

plt.tight_layout()
plt.savefig("monitoring_dashboard.png")
print(f"Total predictions logged: {len(df)}")