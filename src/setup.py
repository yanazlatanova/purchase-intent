from ucimlrepo import fetch_ucirepo 
import pandas as pd
from pathlib import Path
  
# fetch dataset 
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
# data (as pandas dataframes) 
X = online_shoppers_purchasing_intention_dataset.data.features 
y = online_shoppers_purchasing_intention_dataset.data.targets 
  
# metadata 
print(online_shoppers_purchasing_intention_dataset.metadata) 
  
# variable information 
print(online_shoppers_purchasing_intention_dataset.variables)

print(X)

# Join features and targets into a single DataFrame
# If `y` is a DataFrame or Series, concat will align columns correctly
df = pd.concat([X, y], axis=1)

# Shuffle and split randomly in half using seed=1
df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
n = len(df_shuffled)
half = n // 2
part1 = df_shuffled.iloc[:half]
part2 = df_shuffled.iloc[half:]

# Ensure data directory exists (project root/data)
output_dir = Path(__file__).resolve().parents[1] / 'data'
output_dir.mkdir(parents=True, exist_ok=True)

# Save to CSVs
part1.to_csv(output_dir / 'online_shoppers_part1.csv', index=False)
part2.to_csv(output_dir / 'online_shoppers_part2.csv', index=False)

print(f"Saved {len(part1)} rows to {output_dir / 'online_shoppers_part1.csv'}")
print(f"Saved {len(part2)} rows to {output_dir / 'online_shoppers_part2.csv'}")