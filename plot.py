#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file.
df = pd.read_csv("results.csv")

# Convert columns to numeric if they aren't already.
df["K"] = pd.to_numeric(df["K"], errors='coerce')
df["AverageTimeMicroseconds"] = pd.to_numeric(df["AverageTimeMicroseconds"], errors='coerce')

# Create a figure.
plt.figure(figsize=(10, 6))

# Plot each version's data.
for version in df["Version"].unique():
    sub_df = df[df["Version"] == version]
    plt.plot(sub_df["K"], sub_df["AverageTimeMicroseconds"], marker='o', linestyle='-', label=version)

# Set y-axis to logarithmic scale.
plt.yscale("log")

# Label the axes and add title.
plt.xlabel("K (Number of Clusters)")
plt.ylabel("Average Execution Time (microseconds, log scale)")
plt.title("K-Means Execution Time vs. Number of Clusters (1567 x 590)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()

# Save the figure (optional)
plt.savefig("execution_times.png")

# Display the plot.
plt.show()