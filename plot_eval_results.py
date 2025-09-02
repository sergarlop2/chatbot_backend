import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Eval dataset path
EVAL_DATASET = "datasets/eval_dataset.csv"

# Load eval CSV
df = pd.read_csv(EVAL_DATASET, sep=";")

# Global accuracy
accuracy = df["is_correct"].mean()

# Show summary
print("\n---------- SUMMARY ----------\n")
print(df.columns)
print(df.head())
print(df.tail())
print(df.iloc[-1])
print(f"\nGlobal accuracy: {accuracy:.4f}")

# Number of questions per topic
topics = df["topic"].drop_duplicates().tolist()
topic_counts = df["topic"].value_counts()[topics]

print("\nNumber of questions per topic:")
print(topic_counts.to_dict())

# Plot number of questions per topic
plt.figure(1, figsize=(9, 6))
colors = sns.color_palette("viridis", len(topics))
bars = plt.bar(topics, topic_counts, color=colors, edgecolor="black")
plt.title("Number of Questions per Topic", fontsize=16, weight='bold')
plt.ylabel("Number of Questions", fontsize=12)
plt.ylim(top=topic_counts.max() * 1.1)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# Accuracy per topic
topic_accuracy = df.groupby("topic", sort=False)["is_correct"].mean().reindex(topics).round(4)

print("\nAccuracy per topic:")
print(topic_accuracy.to_dict())

# Plot accuracy
plt.figure(2, figsize=(9, 6))
palette = sns.color_palette("Spectral", 256)
colors = [palette[int(acc * 255)] for acc in topic_accuracy.values]
bars = plt.bar(topics, topic_accuracy.values, color=colors, edgecolor="black")
plt.title("Accuracy per Topic", fontsize=16, weight='bold')
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add value labels on bars
for bar, acc in zip(bars, topic_accuracy.values):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.2f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

