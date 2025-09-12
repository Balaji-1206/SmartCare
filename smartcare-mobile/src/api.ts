// src/api.ts
export async function apiGet<T>(base: string, path: string): Promise<T> {
  const r = await fetch(`${base}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
export async function apiPost<T>(base: string, path: string, body: any): Promise<T> {
  const r = await fetch(`${base}${path}`, {
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

// types (simplified)
export type StatusLevel = "GREEN" | "YELLOW" | "RED";
export type Syndrome = { syndrome: string; prob: number; rank: number };
export type AlertT = {
  type: "stockout_risk" | "reorder";
  severity: "HIGH" | "MEDIUM" | "LOW";
  message: string;
  item_code: string;
};
// add this near the other exported types in src/api.ts
export type InventoryRow = {
  name: string;
  on_hand: number;
  reorder_point: number;
};

export type DemandItem = { item_code: string; yhat: number; p10: number; p90: number };
export type MobileToday = {
  for_date: string;
  expected_patients: number;
  delta_vs_yesterday_pct: number | null;
  status: { level: StatusLevel; reason: string };
  top_syndromes: Syndrome[];
  critical_alerts: AlertT[];
  demand_preview: DemandItem[];
  nurse_log_today?: Record<string, any>;
  
};
