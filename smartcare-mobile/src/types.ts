// smartcare-mobile/src/types.ts

export type StatusLevel = "GREEN" | "YELLOW" | "RED";

export interface StatusInfo {
  level: StatusLevel;
  reason: string;
}

export interface Syndrome {
  syndrome: string;
  prob: number;
  rank: number;
}

export interface AlertT {
  type: "stockout_risk" | "reorder";
  severity: "HIGH" | "MEDIUM" | "LOW";
  message: string;
  item_code: string;
}

export interface DemandPreviewItem {
  item_code: string;
  yhat: number;
  p10?: number;
  p90?: number;
}

export interface InventoryRow {
  name: string;
  on_hand: number;
  reorder_point: number;
}

export interface MobileToday {
  for_date: string;
  expected_patients: number;
  delta_vs_yesterday_pct: number | null;
  status: StatusInfo;
  top_syndromes: Syndrome[];
  critical_alerts: AlertT[];
  demand_preview: DemandPreviewItem[];
  nurse_log_today?: Record<string, any>;
}

export interface NurseLogReq {
  date?: string; // "YYYY-MM-DD"
  fever?: number;
  cough?: number;
  diarrhea?: number;
  vomiting?: number;
  cold?: number;
  others?: number;
  notes?: string;
  by?: string;
}

export interface InventoryUpsertReq {
  item_code: string;
  name?: string;
  on_hand?: number;
  reorder_point?: number;
}

export interface WeatherData {
  date: string;
  temperature: number | null;
  rainfall: number | null;
  humidity: number | null;
}
