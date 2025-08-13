# v6_harvest_misses.py
import pandas as pd
from pathlib import Path

SRC = "v6_ollama_simulation.csv"
OUT = "hard_negatives_v6.csv"

df = pd.read_csv(SRC)
misses = df[df["blue_prediction"] == "SAFE"].copy()

if misses.empty:
    print("✅ No misses to harvest. You're fully blocking in this run.")
else:
    harvested = misses[["red_team_output_redacted"]].rename(
        columns={"red_team_output_redacted": "text"}
    )
    harvested["label"] = 1
    harvested.to_csv(OUT, index=False)
    print(f"✅ Harvested {len(harvested)} hard negatives -> {OUT}")
