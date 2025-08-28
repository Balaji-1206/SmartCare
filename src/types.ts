// src/types.ts
export type StatusLevel = "GREEN" | "YELLOW" | "RED";

export interface Syndrome {
  syndrome: string;
  prob: number;
  rank: number;
}

export interface Alert {
  type: "stockout_risk" | "reorder";
  severity: "HIGH" | "MEDIUM" | "LOW";
  message: string;
  item_code: string;
}

export interface DemandPreviewItem {
  item_code: string;
  yhat: number;
  p10: number;
  p90: number;
}

export interface MobileToday {
  expected_patients: number;
  delta_vs_yesterday_pct: number | null;
  status: { level: StatusLevel; reason: string };
  top_syndromes: Syndrome[];
  critical_alerts: Alert[];
  demand_preview: DemandPreviewItem[];
  nurse_log_today?: Record<string, any>;
}

export interface NurseLogReq {
  date: string; // "YYYY-MM-DD"
  fever?: number;
  cough?: number;
  diarrhea?: number;
  vomiting?: number;
  cold?: number;
  notes?: string;
  by?: string;
}
