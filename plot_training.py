import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/train_metrics.csv")

plt.figure()
plt.plot(df["update"], df["rmse_g"])
plt.xlabel("PPO update")
plt.ylabel("RMSE(g)")
plt.title("Training RMSE(g)")
plt.tight_layout()
plt.savefig("logs/rmse_g.png", dpi=200)

plt.figure()
plt.plot(df["update"], df["cat_rate_g"])
plt.xlabel("PPO update")
plt.ylabel("Catastrophic rate (|err|>thresh)")
plt.title("Training Catastrophic Rate")
plt.tight_layout()
plt.savefig("logs/cat_rate_g.png", dpi=200)

print("Saved plots to logs/rmse_g.png and logs/cat_rate_g.png")