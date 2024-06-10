import pandas as pd

df = pd.read_csv("stacking_fault/rate_defect_formation_times.csv")

df["Defect_Formation_Time_Minutes"] = df["Defect_Formation_Time_Seconds"] / 60

df["5_Minute_Period"] = (df["Defect_Formation_Time_Minutes"] // 1) * 1

grouped_df = df.groupby("5_Minute_Period").size().reset_index(name="Count")

print(grouped_df)
