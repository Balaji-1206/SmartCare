// src/constants.ts
import { Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

export const C = {
  bg: "#e5ebf1",
  card: "#ffffff",
  text: "#1f2937",
  sub: "#6b7280",
  chip: "#f1f5f9",
  border: "#cbd5e1",
  primary: "#229ED9",
  green: "#22c55e",
  yellow: "#facc15",
  red: "#ef4444",
  shadow: "rgba(0,0,0,0.08)",
};

export function formatLabel(text: string) {
  if (!text) return text;
  return text.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export const K_API = "smartcare_api_base";
export const K_NURSE = "smartcare_nurse_name";
export const K_AUTHED = "smartcare_authed";

export const API_DEFAULT = Platform.select({
  android: "http://10.0.2.2:8000",
  ios: "http://127.0.0.1:8000",
  default: "http://127.0.0.1:8000",
})!;

export async function getApiBase(fallback = API_DEFAULT) {
  try {
    return (await AsyncStorage.getItem(K_API)) || fallback;
  } catch {
    return fallback;
  }
}
export async function setApiBase(v: string) {
  try {
    await AsyncStorage.setItem(K_API, v);
  } catch {}
}
export async function getNurseName() {
  try {
    return (await AsyncStorage.getItem(K_NURSE)) || "";
  } catch {
    return "";
  }
}
export async function setNurseName(v: string) {
  try {
    await AsyncStorage.setItem(K_NURSE, v);
  } catch {}
}
export async function setAuthed(v: boolean) {
  try {
    await AsyncStorage.setItem(K_AUTHED, v ? "1" : "");
  } catch {}
}
export async function isAuthed() {
  try {
    return (await AsyncStorage.getItem(K_AUTHED)) === "1";
  } catch {
    return false;
  }
}
