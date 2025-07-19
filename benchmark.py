#!/usr/bin/env python3
import subprocess
import csv

# List of executable names
executables = [
    
    "kmeans-serial",
    "kmeans-gpu-v1",
    "kmeans-gpu-v2",
    "kmeans-gpu-v3"
]

dataset = "datasets/dataset7.txt"
results = []  # will store rows in the form [Version, K, AverageTimeMicroseconds]

for exe in executables:
    # Construct the command: pipe dataset to the executable.
    cmd = f"cat {dataset} | ./{exe}"
    print(f"Running {exe}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {exe}: {result.stderr}")
        continue

    # Parse output lines.
    lines = result.stdout.splitlines()
    header_found = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("K,"):
            # Found header; skip it (or store it if you want to check consistency)
            header_found = True
            continue
        if header_found:
            # Each valid line is expected to be: K_value,AverageTimeMicroseconds
            parts = line.split(',')
            if len(parts) >= 2:
                # Prepend the executable version to the row.
                results.append([exe] + parts)

# Write the combined results to a CSV file.
output_file = "results.csv"
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row.
    writer.writerow(["Version", "K", "AverageTimeMicroseconds"])
    for row in results:
        writer.writerow(row)

print(f"CSV written to {output_file}")