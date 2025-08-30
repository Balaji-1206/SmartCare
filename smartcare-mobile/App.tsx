// App.tsx
import "react-native-gesture-handler";
import { Ionicons } from "@expo/vector-icons";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Platform,
  SafeAreaView,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  FlatList,
  RefreshControl,
  Pressable,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

/* =====================
   THEME (Hospital Feel)
===================== */
const C = {
  bg: "#e5ebf1",        // Telegram light gray background
  card: "#ffffff",      // White cards
  text: "#1f2937",      // Dark gray text (almost black)
  sub: "#6b7280",       // Subtle gray for secondary text
  chip: "#f1f5f9",      // Soft bluish-gray chip
  border: "#cbd5e1",    // Light gray border
  primary: "#229ED9",   // Telegram Blue
  green: "#22c55e",     // Success green
  yellow: "#facc15",    // Warning yellow
  red: "#ef4444",       // Danger red
  shadow: "rgba(0,0,0,0.08)", // Soft shadows for cards
};

function formatLabel(text: string): string {
  if (!text) return text;
  return text
    .replace(/_/g, " ")        // replace underscores with spaces
    .replace(/\b\w/g, c => c.toUpperCase()); // capitalize each word
}

/* =====================
   STORAGE KEYS + HELPERS
===================== */
const K_API = "smartcare_api_base";
const K_NURSE = "smartcare_nurse_name";
const K_AUTHED = "smartcare_authed"; // "1" or ""

const API_DEFAULT = Platform.select({
  android: "http://10.0.2.2:8000",
  ios: "http://127.0.0.1:8000",
  default: "http://127.0.0.1:8000",
})!;

async function getApiBase(fallback = API_DEFAULT) {
  try {
    return (await AsyncStorage.getItem(K_API)) || fallback;
  } catch {
    return fallback;
  }
}
async function setApiBase(v: string) {
  try {
    await AsyncStorage.setItem(K_API, v);
  } catch {}
}
async function getNurseName() {
  try {
    return (await AsyncStorage.getItem(K_NURSE)) || "";
  } catch {
    return "";
  }
}
async function setNurseName(v: string) {
  try {
    await AsyncStorage.setItem(K_NURSE, v);
  } catch {}
}
async function setAuthed(v: boolean) {
  try {
    await AsyncStorage.setItem(K_AUTHED, v ? "1" : "");
  } catch {}
}
async function isAuthed() {
  try {
    return (await AsyncStorage.getItem(K_AUTHED)) === "1";
  } catch {
    return false;
  }
}

/* =====================
   SHARED UI (Polished)
===================== */
const Card: React.FC<{ title?: string; children: React.ReactNode; accentLeft?: string }> = ({
  title,
  children,
  accentLeft,
}) => (
  <View
    style={{
      backgroundColor: C.card,
      padding: 16,
      borderRadius: 16,
      marginTop: 14,
      borderLeftWidth: accentLeft ? 6 : 0,
      borderLeftColor: accentLeft || "transparent",
      shadowColor: C.shadow,
      shadowOpacity: 0.15,
      shadowRadius: 6,
      shadowOffset: { width: 0, height: 3 },
      elevation: 3,
    }}
  >
    {title ? (
      <Text style={{ color: C.text, fontWeight: "700", fontSize: 18, marginBottom: 10 }}>
        {title}
      </Text>
    ) : null}
    {children}
  </View>
);

const Chip: React.FC<{ label: string }> = ({ label }) => (
  <View
    style={{
      backgroundColor: C.chip,
      paddingVertical: 6,
      paddingHorizontal: 12,
      borderRadius: 999,
      marginRight: 8,
      marginBottom: 8,
    }}
  >
    <Text style={{ color: C.text, fontSize: 13, fontWeight: "500" }}>{label}</Text>
  </View>
);

const Button: React.FC<{ title: string; onPress: () => void; disabled?: boolean }> = ({
  title,
  onPress,
  disabled,
}) => (
  <TouchableOpacity
    disabled={disabled}
    onPress={onPress}
    style={{
      backgroundColor: disabled ? "#93c5fd" : C.primary,
      paddingVertical: 14,
      borderRadius: 12,
      alignItems: "center",
      opacity: disabled ? 0.6 : 1,
      shadowColor: C.shadow,
      shadowOpacity: 0.25,
      shadowRadius: 5,
      shadowOffset: { width: 0, height: 2 },
      elevation: 2,
    }}
  >
    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 16 }}>{title}</Text>
  </TouchableOpacity>
);

function statusColor(level?: StatusLevel): string {
  if (level === "GREEN") return C.green;
  if (level === "YELLOW") return C.yellow;
  if (level === "RED") return C.red;
  return C.sub;
}

const Pill = ({ text, bg }: { text: string; bg: string }) => (
  <View style={{ backgroundColor: bg, paddingVertical: 5, paddingHorizontal: 12, borderRadius: 999 }}>
    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 12 }}>{text}</Text>
  </View>
);

const ErrorBanner = ({ msg }: { msg: string | null }) =>
  !msg ? null : (
    <View
      style={{
        backgroundColor: "#fee2e2",
        padding: 12,
        borderRadius: 12,
        marginTop: 10,
        borderWidth: 1,
        borderColor: "#fca5a5",
      }}
    >
      <Text style={{ color: "#991b1b", fontWeight: "600" }}>{msg}</Text>
    </View>
  );

const Skeleton = ({ h = 18, w = "100%", mt = 8, br = 8 }: any) => (
  <View
    style={{
      height: h,
      width: w as any,
      marginTop: mt,
      borderRadius: br,
      backgroundColor: "#e5e7eb",
    }}
  />
);

/* =====================
   TYPES (API contracts)
===================== */
/* =====================
   TYPES (API contracts)
===================== */
type StatusLevel = "GREEN" | "YELLOW" | "RED";
type AlertT = {
  type: "stockout_risk" | "reorder";
  severity: "HIGH" | "MEDIUM" | "LOW";
  message: string;
  item_code: string;
};
type Syndrome = { syndrome: string; prob: number; rank: number };
type DemandItem = { item_code: string; yhat: number; p10: number; p90: number };
type MobileToday = {
  for_date: string;
  expected_patients: number;
  delta_vs_yesterday_pct: number | null;
  status: { level: StatusLevel; reason: string };
  top_syndromes: Syndrome[];
  critical_alerts: AlertT[];
  demand_preview: DemandItem[];
  nurse_log_today?: Record<string, any>;
};
type InventoryRow = { name: string; on_hand: number; reorder_point: number };

/* =====================
   SIMPLE API HELPERS
===================== */
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

/* =====================
   SCREENS
===================== */
function formatAlertMessage(msg: string): string {
  // Split into item and details
  const [item, rest] = msg.split(":");

  // Capitalize first letter + replace underscores with spaces
  const prettyItem =
    item.charAt(0).toUpperCase() + item.slice(1).replace(/_/g, " ");

  let prettyRest = rest?.trim() || "";

  // Replace "p90 X" with "predicted X needed"
  prettyRest = prettyRest.replace(/p90\s+(\d+)/i, "predicted $1 needed");

  return `${prettyItem}: ${prettyRest}`;
}


// 1) LOGIN
function LoginScreen({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [api, setApi] = useState(API_DEFAULT);
  const [name, setName] = useState("");

  useEffect(() => {
    (async () => {
      setApi(await getApiBase(API_DEFAULT));
      setName((await getNurseName()) || "");
    })();
  }, []);

  async function testApi() {
    try {
      const r = await fetch(`${api}/`);
      const j = await r.json();
      Alert.alert("SmartCare", `✅ API reachable: ${j?.app ?? "online"}`);
    } catch {
      Alert.alert("SmartCare", "❌ API not reachable. Check URL & server.");
    }
  }

  async function onLogin() {
    if (!name.trim()) {
      Alert.alert("SmartCare", "⚠️ Please enter your name");
      return;
    }
    await setApiBase(api);
    await setNurseName(name.trim());
    await setAuthed(true);
    onLoggedIn();
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <View style={{ padding: 24 }}>
        <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 16 }}>
          <Ionicons name="medkit-outline" size={32} color={C.primary} style={{ marginRight: 8 }} />
          <Text style={{ color: C.text, fontSize: 28, fontWeight: "800" }}>SmartCare</Text>
        </View>
        <Text style={{ color: C.sub, fontSize: 15, marginBottom: 24 }}>
          Sign in to continue
        </Text>

        <Card title="API Base URL">
          <Text style={{ color: C.sub, marginBottom: 6, fontSize: 13 }}>
            Change this if API runs on another device/LAN IP.
          </Text>
          <TextInput
            value={api}
            onChangeText={setApi}
            autoCapitalize="none"
            placeholder={API_DEFAULT}
            placeholderTextColor={C.sub}
            style={{
              backgroundColor: "#f3f4f6",
              color: C.text,
              padding: 12,
              borderRadius: 12,
              borderWidth: 1,
              borderColor: C.border,
              fontSize: 15,
            }}
          />
          <View style={{ marginTop: 12 }}>
            <Button title="Test API" onPress={testApi} />
          </View>
        </Card>

        <Card title="Your Details">
          <Text style={{ color: C.sub, marginBottom: 6, fontSize: 13 }}>Nurse Name</Text>
          <TextInput
            value={name}
            onChangeText={setName}
            placeholder="Meena"
            placeholderTextColor={C.sub}
            style={{
              backgroundColor: "#f3f4f6",
              color: C.text,
              padding: 12,
              borderRadius: 12,
              borderWidth: 1,
              borderColor: C.border,
              fontSize: 15,
            }}
          />
        </Card>

        <View style={{ marginTop: 20 }}>
          <Button title="Login" onPress={onLogin} />
        </View>
      </View>
    </SafeAreaView>
  );
}

// 2) TODAY / HOME
function HomeScreen() {
  const [API_BASE, setAPIBase] = useState(API_DEFAULT);
  const [data, setData] = useState<MobileToday | null>(null);
  const [loading, setLoading] = useState(false);
  const [fever, setFever] = useState("");
  const [cold, setCold] = useState("");
  const [cough, setCough] = useState("");
  const [vomiting, setVomiting] = useState("");
  const [diarrhea, setDiarrhea] = useState("");
  const [others, setOthers] = useState("");
  const [notes, setNotes] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => setAPIBase(await getApiBase(API_DEFAULT)))();
  }, []);

  const load = useCallback(async () => {
    try {
      setErr(null);
      setLoading(true);
      const d = await apiGet<MobileToday>(API_BASE, "/mobile/today");
      setData(d);
    } catch (e: any) {
      const msg = e?.message ?? "Failed to load";
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    load();
  }, [load]);

const submitLog = async () => {
  try {
    const nurse = (await getNurseName()) || "Nurse";
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = `${today.getMonth() + 1}`.padStart(2, "0");
    const dd = `${today.getDate()}`.padStart(2, "0");
    const date = `${yyyy}-${mm}-${dd}`;

    await apiPost(API_BASE, "/nurse/log", {
      date,
      fever: fever ? Number(fever) : undefined,
      cough: cough ? Number(cough) : undefined,
      cold: cold ? Number(cold) : undefined,
      diarrhea: diarrhea ? Number(diarrhea) : undefined,
      vomiting: vomiting ? Number(vomiting) : undefined,
      others: others ? Number(others) : undefined,
      notes: notes || undefined,
      by: nurse,
    });

    // clear inputs
    setFever("");
    setCough("");
    setCold("");
    setDiarrhea("");
    setVomiting("");
    setOthers("");
    setNotes("");

    await load();
    Alert.alert("SmartCare", "✅ Log saved for today");
  } catch (e: any) {
    Alert.alert("SmartCare", e?.message ?? "Failed to save");
  }
};


  const deltaText = useMemo(() => {
    if (data?.delta_vs_yesterday_pct == null) return "—";
    const sign = data.delta_vs_yesterday_pct >= 0 ? "📈" : "📉";
    return `${sign} ${Math.abs(data.delta_vs_yesterday_pct).toFixed(1)}% vs yesterday`;
  }, [data?.delta_vs_yesterday_pct]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <FlatList
        data={[]}
        renderItem={null as any}
        keyExtractor={() => "_"}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={C.primary} />}
        ListHeaderComponent={
          <View style={{ padding: 16 }}>
            {/* Header */}
            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
              <View style={{ flexDirection: "row", alignItems: "center" }}>
                <Ionicons name="pulse-outline" size={24} color={C.primary} style={{ marginRight: 6 }} />
                <Text style={{ color: C.text, fontSize: 22, fontWeight: "800" }}>Today</Text>
              </View>
              <Text
                style={{ color: C.sub, fontSize: 12 }}
                onLongPress={() => {
                  Alert.prompt?.(
                    "API Base",
                    "Change API base URL",
                    async (val) => {
                      if (!val) return;
                      setAPIBase(val);
                      await setApiBase(val);
                      load();
                    },
                    "plain-text",
                    API_BASE
                  );
                }}
              >
                ⚙️ API
              </Text>
            </View>

            {/* Error (if any) */}
            <ErrorBanner msg={err} />

            {/* Expected Patients */}
           <Card>
  <Text style={{ color: C.sub }}>Expected Patients</Text>
  <Text
    style={{
      color: C.text,
      fontSize: 56,
      fontWeight: "900",
      marginTop: 2,
    }}
  >
    {data?.expected_patients ?? "—"}
  </Text>

  {/* 🔹 Show prediction date */}
  <Text style={{ color: C.sub, marginTop: 4, fontSize: 13 }}>
    For {data?.for_date || "—"}
  </Text>

  <Text style={{ color: C.sub, marginTop: 6 }}>{deltaText}</Text>
</Card>



                        {/* Traffic Light */}
            <Card accentLeft={statusColor(data?.status?.level)}>
              <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
                <Text style={{ color: C.text, fontWeight: "800" }}>Status</Text>
                {!!data?.status?.level && (
                  <Pill text={data.status.level} bg={statusColor(data.status.level)} />
                )}
              </View>
              <Text style={{ color: C.sub, marginTop: 6 }}>{data?.status?.reason ?? ""}</Text>
            </Card>

            {/* Top Syndromes */}
            <Card title="Today’s Focus">
  <View style={{ flexDirection: "row", flexWrap: "wrap" }}>
    {(data?.top_syndromes || []).map((s) => (
      <Chip key={s.syndrome} label={formatLabel(s.syndrome)} />
    ))}
  </View>
</Card>


            {/* Critical Alerts */}
            <Card title="Critical Alerts">
              {(data?.critical_alerts?.length ?? 0) === 0 ? (
                <Text style={{ color: C.sub }}>No alerts 🎉</Text>
              ) : (
                <View>
                  {(data?.critical_alerts || []).map((a, idx) => (
                    <View
                      key={`${a.item_code}-${idx}`}
                      style={{
                        backgroundColor: "#fef2f2",
                        padding: 12,
                        borderRadius: 12,
                        marginBottom: 8,
                        borderWidth: 1,
                        borderColor: "#fecaca",
                      }}
                    >
                      <Text style={{ color: C.text }}>{formatAlertMessage(a.message)}</Text>

                      <Text style={{ color: "#b91c1c", marginTop: 2 }}>{a.severity}</Text>
                    </View>
                  ))}
                </View>
              )}
            </Card>

            {/* Demand Preview */}
           <Card title="Demand Preview">
  {(data?.demand_preview || []).map((d) => (
    <View
      key={d.item_code}
      style={{
        flexDirection: "row",
        justifyContent: "space-between",
        paddingVertical: 6,
      }}
    >
     <Text style={{ color: C.text, fontWeight: "600" }}>
  {({
    antibiotics: "Antibiotics",
    malaria_kits: "Malaria Kits",
    ors_packets: "ORS Packets"
  } as Record<string, string>)[d.item_code] || d.item_code}
</Text>

      <Text style={{ color: C.primary, fontWeight: "700" }}>
        ~{Math.round(d.yhat)}
      </Text>
    </View>
  ))}
</Card>


            {/* Nurse Log */}
            <Card title="Log Symptoms">
              <View style={{ flexDirection: "row", gap: 15 }}>
                <TextInput
                  value={fever}
                  onChangeText={setFever}
                  keyboardType="number-pad"
                  placeholder="Fever count"
                  placeholderTextColor={C.sub}
                  style={{
                    flex: 1,
                    backgroundColor: "#f3f4f6",
                    color: C.text,
                    padding: 10,
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: C.border,
                  }}
                />
                <TextInput
                  value={cold}
                  onChangeText={setCold}
                  keyboardType="number-pad"
                  placeholder="Cold count"
                  placeholderTextColor={C.sub}
                  style={{
                    flex: 1,
                    backgroundColor: "#f3f4f6",
                    color: C.text,
                    padding: 10,
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: C.border,
                  }}
                />
              </View>

              <View style={{ flexDirection: "row", gap: 15, marginTop: 10 }}>
                <TextInput
                  value={vomiting}
                  onChangeText={setVomiting}
                  keyboardType="number-pad"
                  placeholder="Vomiting count"
                  placeholderTextColor={C.sub}
                  style={{
                    flex: 1,
                    backgroundColor: "#f3f4f6",
                    color: C.text,
                    padding: 10,
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: C.border,
                  }}
                />
                <TextInput
                  value={diarrhea}
                  onChangeText={setDiarrhea}
                  keyboardType="number-pad"
                  placeholder="Diarrhea count"
                  placeholderTextColor={C.sub}
                  style={{
                    flex: 1,
                    backgroundColor: "#f3f4f6",
                    color: C.text,
                    padding: 10,
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: C.border,
                  }}
                />
              </View>

              <View style={{ flexDirection: "row", gap: 15, marginTop: 10 }}>
                <TextInput
                  value={cough}
                  onChangeText={setCough}
                  keyboardType="number-pad"
                  placeholder="Cough count"
                  placeholderTextColor={C.sub}
                  style={{
                    flex: 1,
                    backgroundColor: "#f3f4f6",
                    color: C.text,
                    padding: 10,
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: C.border,
                  }}
                />
                <TextInput
                  value={others}
                  onChangeText={setOthers}
                  keyboardType="number-pad"
                  placeholder="Others"
                  placeholderTextColor={C.sub}
                  style={{
                    flex: 1,
                    backgroundColor: "#f3f4f6",
                    color: C.text,
                    padding: 10,
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: C.border,
                  }}
                />
</View>
<View style={{ flexDirection: "row", gap: 15, marginTop: 10 }}>
                <TextInput
                  value={notes}
                  onChangeText={setNotes}
                  placeholder="Notes"
                  placeholderTextColor={C.sub}
                  style={{
                    flex: 1,
                    backgroundColor: "#f3f4f6",
                    color: C.text,
                    padding: 10,
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: C.border,
                  }}
                />
              </View>
              <View style={{ marginTop: 14 }}>
                <Button
                  title="Save Log"
                  onPress={submitLog}
                  disabled={!fever && !cough && !notes && !cold && !vomiting && !diarrhea && !others}
                />
              </View>
            </Card>

            {/* Today’s nurse log snapshot */}
            <Card title="Today’s Log (saved)">
              {data?.nurse_log_today && Object.keys(data.nurse_log_today).length > 0 ? (
                <View style={{ gap: 6 }}>
                  {data.nurse_log_today.date && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="calendar-outline" size={18} color={C.primary} style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text, fontWeight: "700" }}>{data.nurse_log_today.date}</Text>
                    </View>
                  )}
                  {data.nurse_log_today.fever != null && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="thermometer-outline" size={18} color={C.red} style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text }}>Fever - {data.nurse_log_today.fever}</Text>
                    </View>
                  )}
                  {data.nurse_log_today.cough != null && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="leaf-outline" size={18} color={C.yellow} style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text }}>Cough - {data.nurse_log_today.cough}</Text>
                    </View>
                  )}
                  {data.nurse_log_today.diarrhea != null && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="water-outline" size={18} color="#3b82f6" style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text }}>Diarrhea - {data.nurse_log_today.diarrhea}</Text>
                    </View>
                  )}
                  {data.nurse_log_today.vomiting != null && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="alert-circle-outline" size={18} color={C.red} style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text }}>Vomiting - {data.nurse_log_today.vomiting}</Text>
                    </View>
                  )}
                  {data.nurse_log_today.cold != null && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="snow-outline" size={18} color="#06b6d4" style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text }}>Cold - {data.nurse_log_today.cold}</Text>
                    </View>
                  )}
                  {data.nurse_log_today.others != null && (
  <View style={{ flexDirection: "row", alignItems: "center" }}>
    <Ionicons name="medical-outline" size={18} color={C.sub} style={{ marginRight: 6 }} />
    <Text style={{ color: C.text }}>Others - {data.nurse_log_today.others}</Text>
  </View>
)}
                  {data.nurse_log_today.notes && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="document-text-outline" size={18} color={C.sub} style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text }}>Notes - {data.nurse_log_today.notes}</Text>
                    </View>
                  )}
                  {data.nurse_log_today.by && (
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <Ionicons name="person-circle-outline" size={18} color={C.green} style={{ marginRight: 6 }} />
                      <Text style={{ color: C.text }}>By - {data.nurse_log_today.by}</Text>
                    </View>
                  )}
                </View>
              ) : (
                <Text style={{ color: C.sub }}>Nothing logged yet</Text>
              )}
            </Card>

            <View style={{ height: 80 }} />
          </View>
        }
      />
    </SafeAreaView>
  );
}

// 3) ALERTS
function AlertsScreen({ navigation }: any) {
  const [api, setApi] = useState(API_DEFAULT);
  const [alerts, setAlerts] = useState<AlertT[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<"ALL" | "HIGH" | "MEDIUM" | "LOW">("ALL");

  useEffect(() => {
    (async () => setApi(await getApiBase(API_DEFAULT)))();
  }, []);

  const load = useCallback(async () => {
  try {
    setLoading(true);
    const j = await apiGet<{ alerts: AlertT[] }>(api, "/alerts");  // ✅ new endpoint
    setAlerts(j.alerts || []);
  } catch (e: any) {
    Alert.alert("SmartCare", e?.message ?? "Failed to load alerts");
  } finally {
    setLoading(false);
  }
}, [api]);


  useEffect(() => {
    load();
  }, [load]);

  const shown = useMemo(
    () => (filter === "ALL" ? alerts : alerts.filter((a) => a.severity === filter)),
    [alerts, filter]
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <ScrollView
        contentContainerStyle={{ padding: 16 }}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={C.primary} />}
      >
        <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 12 }}>
          <Ionicons name="alert-circle-outline" size={24} color={C.red} style={{ marginRight: 6 }} />
          <Text style={{ color: C.text, fontSize: 22, fontWeight: "800" }}>Alerts</Text>
        </View>

        <View style={{ flexDirection: "row", gap: 8, marginTop: 4 }}>
          {(["ALL", "HIGH", "MEDIUM", "LOW"] as const).map((f) => (
            <Pressable
              key={f}
              onPress={() => setFilter(f)}
              style={{
                backgroundColor: filter === f ? C.primary : "#f3f4f6",
                paddingVertical: 6,
                paddingHorizontal: 12,
                borderRadius: 999,
              }}
            >
              <Text style={{ color: filter === f ? "#fff" : C.text, fontWeight: "600" }}>{f}</Text>
            </Pressable>
          ))}
        </View>

        <View style={{ marginTop: 16 }}>
          {shown.length === 0 ? (
            <Text style={{ color: C.sub }}>No alerts 🎉</Text>
          ) : (
            shown.map((a, i) => (
              <View
                key={i}
                style={{
                  backgroundColor: C.card,
                  padding: 14,
                  borderRadius: 14,
                  marginBottom: 10,
                  borderWidth: 1,
                  borderColor: C.border,
                }}
              >
                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.text, fontWeight: "700" }}>
                    {a.type === "stockout_risk" ? "Stock-out risk" : "Reorder"}
                  </Text>
                  <Pill
                    text={a.severity}
                    bg={a.severity === "HIGH" ? C.red : a.severity === "MEDIUM" ? C.yellow : C.green}
                  />
                </View>
                <Text style={{ color: C.sub, marginTop: 6 }}>{a.message}</Text>
                <Pressable onPress={() => navigation.navigate("Inventory", { focus: a.item_code })} style={{ marginTop: 8 }}>
                  <Text style={{ color: C.primary, fontWeight: "600" }}>Open “{a.item_code}” in Inventory →</Text>
                </Pressable>
              </View>
            ))
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// 4) INVENTORY
function InventoryScreen({ route }: any) {
  const focusCode: string | undefined = route?.params?.focus;
  const [API_BASE, setAPIBase] = useState(API_DEFAULT);
  const [inv, setInv] = useState<Record<string, InventoryRow>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => setAPIBase(await getApiBase(API_DEFAULT)))();
  }, []);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiGet<Record<string, InventoryRow>>(API_BASE, "/inventory");
      setInv(data || {});
    } catch (e: any) {
      Alert.alert("SmartCare", e?.message ?? "Failed to load inventory");
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    load();
  }, [load]);

  const updateItem = async (code: string, patch: Partial<InventoryRow>) => {
    try {
      await apiPost(API_BASE, "/inventory/upsert", { item_code: code, ...patch });
      await load();
    } catch (e: any) {
      Alert.alert("SmartCare", e?.message ?? "Failed to update");
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <ScrollView
        contentContainerStyle={{ padding: 16 }}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={C.primary} />}
      >
        <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 12 }}>
          <Ionicons name="cube-outline" size={24} color={C.primary} style={{ marginRight: 6 }} />
          <Text style={{ color: C.text, fontSize: 22, fontWeight: "800" }}>Inventory</Text>
        </View>

        {Object.entries(inv).map(([code, row]) => (
          <Card key={code} title={`${row.name}`}>
            <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
              <Text style={{ color: C.sub }}>On hand</Text>
              <Text style={{ color: C.text, fontWeight: "700" }}>{row.on_hand}</Text>
            </View>
            <View style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 6 }}>
              <Text style={{ color: C.sub }}>Reorder point</Text>
              <Text style={{ color: C.text, fontWeight: "700" }}>{row.reorder_point}</Text>
            </View>
            <View style={{ flexDirection: "row", gap: 8, marginTop: 10 }}>
              <TextInput
                placeholder="Set on-hand"
                placeholderTextColor={C.sub}
                keyboardType="number-pad"
                onSubmitEditing={(e) => updateItem(code, { on_hand: Number(e.nativeEvent.text || 0) })}
                style={{
                  flex: 1,
                  backgroundColor: "#f3f4f6",
                  color: C.text,
                  padding: 10,
                  borderRadius: 10,
                  borderWidth: 1,
                  borderColor: C.border,
                }}
              />
              <TextInput
                placeholder="Set reorder pt"
                placeholderTextColor={C.sub}
                keyboardType="number-pad"
                onSubmitEditing={(e) =>
                  updateItem(code, { reorder_point: Number(e.nativeEvent.text || 0) })
                }
                style={{
                  flex: 1,
                  backgroundColor: "#f3f4f6",
                  color: C.text,
                  padding: 10,
                  borderRadius: 10,
                  borderWidth: 1,
                  borderColor: C.border,
                }}
              />
            </View>
            {focusCode === code ? (
              <Text style={{ color: C.primary, marginTop: 6 }}>(Opened from Alerts)</Text>
            ) : null}
          </Card>
        ))}

        <View style={{ height: 80 }} />
      </ScrollView>
    </SafeAreaView>
  );
}


// 5) SETTINGS
function SettingsScreen({ onLogout }: { onLogout: () => void }) {
  const [api, setApi] = useState(API_DEFAULT);
  const [nurse, setNurse] = useState("");

  useEffect(() => {
    (async () => {
      setApi(await getApiBase(API_DEFAULT));
      setNurse((await getNurseName()) || "");
    })();
  }, []);

  async function save() {
    await setApiBase(api);
    await setNurseName(nurse || "Nurse");
    Alert.alert("SmartCare", "✅ Settings saved");
  }
  async function test() {
    try {
      const r = await fetch(`${api}/`);
      const j = await r.json();
      Alert.alert("SmartCare", `✅ API reachable: ${j?.app ?? "server online"}`);
    } catch {
      Alert.alert("SmartCare", "❌ Cannot reach API");
    }
  }
  async function logout() {
    await setAuthed(false);
    onLogout();
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <ScrollView contentContainerStyle={{ padding: 16 }}>
        <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 16 }}>
          <Ionicons name="settings-outline" size={24} color={C.primary} style={{ marginRight: 6 }} />
          <Text style={{ color: C.text, fontSize: 22, fontWeight: "800" }}>Settings</Text>
        </View>

        <Card title="API Base URL">
          <TextInput
            value={api}
            onChangeText={setApi}
            autoCapitalize="none"
            placeholder={API_DEFAULT}
            placeholderTextColor={C.sub}
            style={{
              backgroundColor: "#f3f4f6",
              color: C.text,
              padding: 12,
              borderRadius: 12,
              borderWidth: 1,
              borderColor: C.border,
            }}
          />
         <View style={{ flexDirection: "row", gap: 12, marginTop: 12 }}>
  <View style={{ flex: 1 }}>
    <Button title="Save" onPress={save} />
  </View>
  <View style={{ flex: 1 }}>
    <Button title="Test API" onPress={test} />
  </View>
</View>

        </Card>

        <Card title="Nurse Name">
          <TextInput
            value={nurse}
            onChangeText={setNurse}
            placeholder="Meena"
            placeholderTextColor={C.sub}
            style={{
              backgroundColor: "#f3f4f6",
              color: C.text,
              padding: 12,
              borderRadius: 12,
              borderWidth: 1,
              borderColor: C.border,
            }}
          />
        </Card>

        <View style={{ marginTop: 20 }}>
          <Button title="Logout" onPress={logout} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

/* =====================
   NAVIGATION
===================== */
const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function MainTabs({ onLogout }: { onLogout: () => void }) {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: { backgroundColor: C.card, borderTopColor: C.border },
        tabBarActiveTintColor: C.primary,
        tabBarInactiveTintColor: C.sub,
        tabBarIcon: ({ color, size }) => {
          const map: Record<string, keyof typeof Ionicons.glyphMap> = {
            Today: "pulse-outline",
            Alerts: "alert-circle-outline",
            Inventory: "cube-outline",
            Settings: "settings-outline",
          };
          const name = map[route.name] ?? "ellipse-outline";
          return <Ionicons name={name} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Today" component={HomeScreen} />
      <Tab.Screen name="Alerts" component={AlertsScreen} />
      <Tab.Screen name="Inventory" component={InventoryScreen} />
      <Tab.Screen name="Settings">
        {() => <SettingsScreen onLogout={onLogout} />}
      </Tab.Screen>
    </Tab.Navigator>
  );
}

/* =====================
   APP WRAPPER
===================== */
export default function App() {
  const [ready, setReady] = useState(false);
  const [authed, setAuthedState] = useState(false);

  useEffect(() => {
    (async () => {
      setAuthedState(await isAuthed());
      setReady(true);
    })();
  }, []);

  const handleLoggedIn = () => setAuthedState(true);
  const handleLoggedOut = () => setAuthedState(false);

  if (!ready) return null;

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!authed ? (
          <Stack.Screen name="Login">
            {() => <LoginScreen onLoggedIn={handleLoggedIn} />}
          </Stack.Screen>
        ) : (
          <Stack.Screen name="Main">
            {() => <MainTabs onLogout={handleLoggedOut} />}
          </Stack.Screen>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
