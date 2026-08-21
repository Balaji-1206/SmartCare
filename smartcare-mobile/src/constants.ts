// src/constants.ts
import { Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

export const C = {
  // Premium Clean Light Medical Palette
  bg: "#f1f5f9",
  bgSubtle: "#f8fafc",
  card: "#ffffff",
  cardElevated: "#ffffff",
  glassBg: "rgba(255, 255, 255, 0.95)",
  glassBorder: "#e2e8f0",
  glassBorderActive: "#38bdf8",
  
  // Typography (High Contrast Dark Slate)
  text: "#0f172a",
  textSecondary: "#334155",
  sub: "#64748b",
  textMuted: "#94a3b8",
  
  // Controls & Inputs
  chip: "#f1f5f9",
  chipActive: "#0284c7",
  inputBg: "#ffffff",
  inputBorder: "#cbd5e1",
  inputFocusBorder: "#0284c7",
  border: "#e2e8f0",
  divider: "#f1f5f9",
  
  // Vibrant Clinical Brand Accents (Sky / Cyan / Blue)
  primary: "#0284c7",
  primaryLight: "#0284c7",
  primaryDark: "#0369a1",
  primaryGlow: "rgba(2, 132, 199, 0.15)",
  primaryBg: "#e0f2fe",
  
  // Status Levels
  green: "#16a34a",
  greenLight: "#15803d",
  greenGlow: "rgba(22, 163, 74, 0.15)",
  greenBg: "#dcfce7",
  
  yellow: "#d97706",
  yellowLight: "#b45309",
  yellowGlow: "rgba(217, 119, 6, 0.15)",
  yellowBg: "#fef3c7",
  
  red: "#dc2626",
  redLight: "#b91c1c",
  redGlow: "rgba(220, 38, 38, 0.15)",
  redBg: "#fee2e2",
  
  purple: "#7c3aed",
  purpleLight: "#6d28d9",
  purpleGlow: "rgba(124, 58, 237, 0.15)",
  purpleBg: "#ede9fe",
  
  cyan: "#0891b2",
  cyanBg: "#cffafe",

  shadow: "rgba(15, 23, 42, 0.08)",
};

export function formatLabel(text: string) {
  if (!text) return text;
  return text.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export const K_API = "smartcare_api_base";
export const K_NURSE = "smartcare_nurse_name";
export const K_AUTHED = "smartcare_authed";
export const K_THEME = "smartcare_theme";
export const K_LAT = "smartcare_weather_lat";
export const K_LON = "smartcare_weather_lon";

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

export async function getLatLon(): Promise<{ lat: string; lon: string }> {
  try {
    const lat = (await AsyncStorage.getItem(K_LAT)) || "";
    const lon = (await AsyncStorage.getItem(K_LON)) || "";
    return { lat, lon };
  } catch {
    return { lat: "", lon: "" };
  }
}

export async function setLatLon(lat: string, lon: string) {
  try {
    await AsyncStorage.setItem(K_LAT, lat);
    await AsyncStorage.setItem(K_LON, lon);
  } catch {}
}
