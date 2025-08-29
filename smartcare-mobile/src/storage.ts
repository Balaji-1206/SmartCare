import AsyncStorage from "@react-native-async-storage/async-storage";

const K_API    = "smartcare_api_base";
const K_NURSE  = "smartcare_nurse_name";
const K_PIN    = "smartcare_pin";
const K_AUTHED = "smartcare_authed"; // "1" or ""

export async function getApiBase(fallback: string) {
  try { return (await AsyncStorage.getItem(K_API)) || fallback; } catch { return fallback; }
}
export async function setApiBase(v: string) { try { await AsyncStorage.setItem(K_API, v); } catch {} }

export async function getNurseName() {
  try { return (await AsyncStorage.getItem(K_NURSE)) || ""; } catch { return ""; }
}
export async function setNurseName(v: string) { try { await AsyncStorage.setItem(K_NURSE, v); } catch {} }

export async function getPin() {
  try { return (await AsyncStorage.getItem(K_PIN)) || ""; } catch { return ""; }
}
export async function setPin(v: string) { try { await AsyncStorage.setItem(K_PIN, v); } catch {} }

export async function setAuthed(v: boolean) {
  try { await AsyncStorage.setItem(K_AUTHED, v ? "1" : ""); } catch {}
}
export async function isAuthed() {
  try { return (await AsyncStorage.getItem(K_AUTHED)) === "1"; } catch { return false; }
}
