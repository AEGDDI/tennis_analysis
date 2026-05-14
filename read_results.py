import pickle
import pandas as pd

with open("data/rg2026_results.pkl", "rb") as f:
    data = pickle.load(f)

results = data["results"]   # DataFrame: one row per player, columns include 'name', 'W', ...
ratings = data["ratings"]   # clay strength scores per player
draw    = data["draw"]      # tournament bracket structure

# All columns available
print(results.columns.tolist())

# Top 10 title favourites
print(results.head(10).to_string(index=False))

# Full sorted table
print(results.sort_values("W", ascending=False).to_string())
