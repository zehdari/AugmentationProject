from pathlib import Path
import pandas as pd
dfs = []
for f in Path("/users/PAS2119/darklord/CVfinalproject/AugmentationProject/csv_all_runs").glob("*.csv"):
    df = pd.read_csv(f)
    df["experiment"] = f.stem
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv("all_experiments.csv", index=False)
