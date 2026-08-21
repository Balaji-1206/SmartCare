// src/api.ts
function joinUrl(base: string, path: string) {
  const b = (base || "").replace(/\/+$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

export async function apiGet<T>(base: string, path: string): Promise<T> {
  const url = joinUrl(base, path);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export async function apiPost<T>(base: string, path: string, body: any): Promise<T> {
  const url = joinUrl(base, path);
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${r.status} ${r.statusText} — ${txt}`);
  }
  return r.json();
}

// ─── Core Types ─────────────────────────────────────────────
export type StatusLevel = "GREEN" | "YELLOW" | "RED";

export type Syndrome = {
  syndrome: string;
  prob: number;
  rank: number;
};

export type AlertT = {
  type: "stockout_risk" | "reorder";
  severity: "HIGH" | "MEDIUM" | "LOW";
  message: string;
  item_code: string;
};

export type InventoryRow = {
  name: string;
  on_hand: number;
  reorder_point: number;
  last_updated?: string;
};

export type DemandItem = {
  item_code: string;
  yhat: number;
  p10: number | null;
  p90: number | null;
};

export type WeatherContext = {
  temperature: number | null;
  rainfall: number | null;
  humidity: number | null;
};

export type NurseLogDay = {
  date: string;
  fever: number;
  cough: number;
  diarrhea: number;
  vomiting: number;
  cold: number;
  others: number;
  notes: string;
  by: string;
  has_entry: boolean;
};

export type MobileToday = {
  for_date: string;
  expected_patients: number;
  expected_patients_p10?: number;
  expected_patients_p90?: number;
  delta_vs_yesterday_pct: number | null;
  last7_volumes?: number[];
  weather?: WeatherContext;
  status: { level: StatusLevel; reason: string };
  top_syndromes: Syndrome[];
  critical_alerts: AlertT[];
  all_alerts_count?: number;
  demand_preview: DemandItem[];
  nurse_log_today?: Record<string, any>;
};

export type StatsSummary = {
  last7: {
    dates: string[];
    volumes: number[];
    mean: number | null;
    max: number | null;
    min: number | null;
    std: number | null;
  };
  last30: {
    mean: number | null;
    max: number | null;
    min: number | null;
    std: number | null;
  };
  day_of_week_avg: Record<string, number>;
  trend_7d_pct: number | null;
  total_records: number;
};

export type NurseLogHistory = {
  days: number;
  history: NurseLogDay[];
};

export type OutbreakAlert = {
  syndrome: string;
  label: string;
  current_7d_cases: number;
  baseline_7d_cases: number;
  ratio: number | null;
  severity: "HIGH" | "MEDIUM";
  days_with_cases: number;
  daily_trend: number[];
  message: string;
  recommendation: string;
};

export type OutbreakResult = {
  outbreak_alerts: OutbreakAlert[];
  has_high_outbreak: boolean;
  total_outbreaks: number;
  checked_at: string;
  window_days: number;
};

export type EnrichedInventoryRow = InventoryRow & {
  days_to_stockout: number | null;
  daily_demand: number | null;
  daily_demand_p10?: number | null;
  daily_demand_p90?: number | null;
};

