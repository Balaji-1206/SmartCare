import React, { useEffect, useMemo, useState } from "react";
import { Platform, SafeAreaView, View, Text, FlatList, TextInput, TouchableOpacity, RefreshControl, Alert } from "react-native";
import { getMobileToday, postNurseLog } from "./src/api";

// =====================
// CONFIG — API BASE URL
// =====================
// Running the FastAPI server locally? Set your base like this:
// - Android emulator: http://10.0.2.2:8000
// - iOS simulator:    http://127.0.0.1:8000
// - Physical device:  http://<YOUR-COMPUTER-LAN-IP>:8000 (e.g., http://192.168.1.7:8000)
const API_BASE_DEFAULT = Platform.select({
  android: "http://10.0.2.2:8000",
  ios: "http://127.0.0.1:8000",
  default: "http://127.0.0.1:8000",
});

// If you want to hardcode your LAN IP, uncomment:
// const API_BASE_DEFAULT = "http://192.168.1.7:8000";

// ===============
// TYPE DEFINITIONS
// ===============
interface Syndrome { syndrome: string; prob: number; rank: number }
interface AlertT { type: string; severity: "HIGH"|"MEDIUM"|"LOW"; message: string; item_code: string }
interface DemandItem { item_code: string; yhat: number; p10: number; p90: number }
interface MobileToday {
  expected_patients: number;
  delta_vs_yesterday_pct: number | null;
  status: { level: "GREEN" | "YELLOW" | "RED"; reason: string };
  top_syndromes: Syndrome[];
  critical_alerts: AlertT[];
  demand_preview: DemandItem[];
  nurse_log_today?: Record<string, any>;
}

// ======
// THEME
// ======
const colors = {
  bg: "#0b1220",
  card: "#111827",
  text: "#ffffff",
  sub: "#9ca3af",
  chip: "#1f2937",
  green: "#22c55e",
  yellow: "#eab308",
  red: "#ef4444",
  border: "#374151",
};

function statusColor(level?: string) {
  switch (level) {
    case "GREEN": return colors.green;
    case "YELLOW": return colors.yellow;
    case "RED": return colors.red;
    default: return colors.sub;
  }
}

// ==============
// SMALL WIDGETS
// ==============
const Card: React.FC<{title?: string; children: React.ReactNode; accentLeft?: string}> = ({ title, children, accentLeft }) => (
  <View style={{ backgroundColor: colors.card, padding: 16, borderRadius: 14, marginTop: 12, borderLeftWidth: accentLeft ? 6 : 0, borderLeftColor: accentLeft || "transparent" }}>
    {title ? <Text style={{ color: colors.text, fontWeight: "700", marginBottom: 8 }}>{title}</Text> : null}
    {children}
  </View>
);

const Chip: React.FC<{label: string}> = ({ label }) => (
  <View style={{ backgroundColor: colors.chip, paddingVertical: 6, paddingHorizontal: 10, borderRadius: 999, marginRight: 8, marginBottom: 8 }}>
    <Text style={{ color: "#e5e7eb" }}>{label}</Text>
  </View>
);

const Button: React.FC<{title: string; onPress: () => void}> = ({ title, onPress }) => (
  <TouchableOpacity onPress={onPress} style={{ backgroundColor: "#2563eb", paddingVertical: 12, borderRadius: 10, alignItems: "center" }}>
    <Text style={{ color: "#fff", fontWeight: "700" }}>{title}</Text>
  </TouchableOpacity>
);

// ==========
// API CLIENT
// ==========
async function apiGet<T>(base: string, path: string): Promise<T> {
  const r = await fetch(`${base}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function apiPost<T>(base: string, path: string, body: any): Promise<T> {
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

// ============
// MAIN SCREEN
// ============
export default function App() {
  const [API_BASE, setAPIBase] = useState<string>(API_BASE_DEFAULT!);
  const [data, setData] = useState<MobileToday | null>(null);
  const [loading, setLoading] = useState(false);
  const [fever, setFever] = useState("");
  const [cough, setCough] = useState("");
  const [notes, setNotes] = useState("");

  const load = async () => {
    try {
      setLoading(true);
      const d = await apiGet<MobileToday>(API_BASE, "/mobile/today");
      setData(d);
    } catch (e: any) {
      Alert.alert("SmartCare", e?.message ?? "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [API_BASE]);

  const submitLog = async () => {
    try {
      const today = new Date();
      const yyyy = today.getFullYear();
      const mm = `${today.getMonth()+1}`.padStart(2, "0");
      const dd = `${today.getDate()}`.padStart(2, "0");
      const date = `${yyyy}-${mm}-${dd}`;

      await apiPost(API_BASE, "/nurse/log", {
        date,
        fever: fever ? Number(fever) : undefined,
        cough: cough ? Number(cough) : undefined,
        notes: notes || undefined,
        by: "Nurse", // TODO: replace with signed-in user name/id
      });
      setFever(""); setCough(""); setNotes("");
      await load();
      Alert.alert("SmartCare", "Log saved for today");
    } catch (e: any) {
      Alert.alert("SmartCare", e?.message ?? "Failed to save");
    }
  };

  const deltaText = useMemo(() => {
    if (data?.delta_vs_yesterday_pct == null) return "—";
    const sign = data.delta_vs_yesterday_pct >= 0 ? "↗" : "↘";
    return `${sign} ${Math.abs(data.delta_vs_yesterday_pct).toFixed(1)}% vs yesterday`;
  }, [data?.delta_vs_yesterday_pct]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <FlatList
        data={[]}
        renderItem={null as any}
        keyExtractor={() => "_"}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor="#fff" />}
        ListHeaderComponent={
          <View style={{ padding: 16 }}>
            {/* Header */}
            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
              <Text style={{ color: colors.text, fontSize: 22, fontWeight: "800" }}>SmartCare — Today</Text>
              {/* Quick API base inline edit for dev convenience */}
              <Text style={{ color: colors.sub, fontSize: 12 }} onLongPress={() => {
                Alert.prompt?.("API Base", "Change API base URL", (val) => val && setAPIBase(val), "plain-text", API_BASE);
              }}>API</Text>
            </View>

            {/* Expected Patients */}
            <Card>
              <Text style={{ color: colors.sub }}>Expected Patients</Text>
              <Text style={{ color: colors.text, fontSize: 56, fontWeight: "900", marginTop: 2 }}>
                {data?.expected_patients ?? "—"}
              </Text>
              <Text style={{ color: colors.sub, marginTop: 6 }}>{deltaText}</Text>
            </Card>

            {/* Traffic Light */}
            <Card accentLeft={statusColor(data?.status?.level)}>
              <Text style={{ color: colors.text, fontWeight: "800" }}>Status: {data?.status?.level ?? "—"}</Text>
              <Text style={{ color: colors.sub, marginTop: 4 }}>{data?.status?.reason ?? ""}</Text>
            </Card>

            {/* Top Syndromes */}
            <Card title="Today’s Focus">
              <View style={{ flexDirection: "row", flexWrap: "wrap" }}>
                {(data?.top_syndromes || []).map((s) => (
                  <Chip key={s.syndrome} label={s.syndrome} />
                ))}
              </View>
            </Card>

            {/* Critical Alerts */}
            <Card title="Critical Alerts">
              {(data?.critical_alerts?.length ?? 0) === 0 ? (
                <Text style={{ color: colors.sub }}>No alerts</Text>
              ) : (
                <View>
                  {(data?.critical_alerts || []).map((a, idx) => (
                    <View key={`${a.item_code}-${idx}`} style={{ backgroundColor: colors.chip, padding: 10, borderRadius: 10, marginBottom: 8, borderWidth: 1, borderColor: colors.border }}>
                      <Text style={{ color: colors.text }}>{a.message}</Text>
                      <Text style={{ color: colors.sub, marginTop: 2 }}>{a.severity}</Text>
                    </View>
                  ))}
                </View>
              )}
            </Card>

            {/* Demand Preview */}
            <Card title="Demand Preview">
              {(data?.demand_preview || []).map((d) => (
                <View key={d.item_code} style={{ flexDirection: "row", justifyContent: "space-between", paddingVertical: 6 }}>
                  <Text style={{ color: colors.text }}>{d.item_code}</Text>
                  <Text style={{ color: colors.sub }}>~{d.yhat.toFixed(2)} (p10 {d.p10.toFixed(2)} / p90 {d.p90.toFixed(2)})</Text>
                </View>
              ))}
            </Card>

            {/* Nurse Log */}
            <Card title="Log Symptoms">
              <View style={{ flexDirection: "row", gap: 8 }}>
                <TextInput
                  value={fever}
                  onChangeText={setFever}
                  keyboardType="number-pad"
                  placeholder="Fever count"
                  placeholderTextColor={colors.sub}
                  style={{ flex: 1, backgroundColor: colors.bg, color: colors.text, padding: 10, borderRadius: 10 }}
                />
                <TextInput
                  value={cough}
                  onChangeText={setCough}
                  keyboardType="number-pad"
                  placeholder="Cough count"
                  placeholderTextColor={colors.sub}
                  style={{ flex: 1, backgroundColor: colors.bg, color: colors.text, padding: 10, borderRadius: 10 }}
                />
              </View>
              <TextInput
                value={notes}
                onChangeText={setNotes}
                placeholder="Notes"
                placeholderTextColor={colors.sub}
                style={{ marginTop: 8, backgroundColor: colors.bg, color: colors.text, padding: 10, borderRadius: 10 }}
              />
              <View style={{ marginTop: 10 }}>
                <Button title="Save Log" onPress={submitLog} />
              </View>
            </Card>

            {/* Today’s nurse log snapshot */}
            <Card title="Today’s Log (saved)">
              {data?.nurse_log_today && Object.keys(data.nurse_log_today).length > 0 ? (
                <Text style={{ color: colors.text }}>{JSON.stringify(data.nurse_log_today)}</Text>
              ) : (
                <Text style={{ color: colors.sub }}>Nothing logged yet</Text>
              )}
            </Card>

            <View style={{ height: 80 }} />
          </View>
        }
      />
    </SafeAreaView>
  );
}
