CREATE TABLE IF NOT EXISTS visits_daily(
  date DATE PRIMARY KEY,
  facility_id TEXT DEFAULT 'C001',
  total_visits INT NOT NULL,
  male_patients INT,
  female_patients INT,
  children_under5 INT,
  temperature NUMERIC,
  rainfall INT,
  humidity INT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_visits_fac_date ON visits_daily(facility_id, date);

CREATE TABLE IF NOT EXISTS demand_daily(
  date DATE,
  facility_id TEXT DEFAULT 'C001',
  item_code TEXT,
  units_used INT,
  PRIMARY KEY (date, facility_id, item_code)
);

CREATE TABLE IF NOT EXISTS inventory(
  facility_id TEXT,
  item_code TEXT,
  name TEXT,
  on_hand INT,
  reorder_point INT DEFAULT 0,
  updated_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (facility_id, item_code)
);

CREATE TABLE IF NOT EXISTS pred_volume_daily(
  date DATE,
  facility_id TEXT,
  yhat NUMERIC,
  p10 NUMERIC,
  p90 NUMERIC,
  status_level TEXT,
  model_ver TEXT,
  PRIMARY KEY (date, facility_id)
);

CREATE TABLE IF NOT EXISTS pred_demand_daily(
  date DATE,
  facility_id TEXT,
  item_code TEXT,
  yhat NUMERIC,
  p10 NUMERIC,
  p90 NUMERIC,
  model_ver TEXT,
  PRIMARY KEY (date, facility_id, item_code)
);

CREATE TABLE IF NOT EXISTS model_metrics(
  date DATE,
  task TEXT,
  metric TEXT,
  value NUMERIC,
  model_ver TEXT,
  PRIMARY KEY (date, task, metric)
);
