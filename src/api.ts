import { API_BASE } from "./config";
import { MobileToday, NurseLogReq } from "./types";

export async function getMobileToday(): Promise<MobileToday> {
  const res = await fetch(`${API_BASE}/mobile/today`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function postNurseLog(payload: NurseLogReq) {
  const res = await fetch(`${API_BASE}/nurse/log`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}
