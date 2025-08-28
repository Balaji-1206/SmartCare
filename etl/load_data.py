import pandas as pd
from sqlalchemy import create_engine
import os

DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/smartcare")
engine = create_engine(DB_URL)

df = pd.read_csv("data/raw/data10yrs.csv", parse_dates=["date"])

# visits_daily
vis = df.rename(columns={
    "total_patients": "total_visits",
    "male_patients": "male_patients",
    "female_patients": "female_patients",
    "temperature": "temperature",
    "rainfall": "rainfall",
    "humidity": "humidity"
})[["date","total_visits","male_patients","female_patients","temperature","rainfall","humidity"]]
vis.to_sql("visits_daily", engine, if_exists="append", index=False)

# demand_daily (map columns ending with _used to item codes)
demand_cols = [c for c in df.columns if c.endswith("_used")]
rows = []
for _, r in df.iterrows():
    for c in demand_cols:
        rows.append({"date": r["date"], "item_code": c.replace("_used",""), "units_used": int(r[c])})
pd.DataFrame(rows).to_sql("demand_daily", engine, if_exists="append", index=False)
