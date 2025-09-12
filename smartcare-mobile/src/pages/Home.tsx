// src/pages/Home.tsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  SafeAreaView,
  View,
  Text,
  ScrollView,
  RefreshControl,
  TextInput,
  TouchableOpacity,
  Alert,
  StyleSheet,
  Animated,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Card, Chip, Button, ErrorBanner, Pill } from "../ui";
import { C, getApiBase, API_DEFAULT } from "../constants";
import { apiGet, apiPost } from "../api";

type MobileToday = {
  for_date: string;
  expected_patients: number;
  delta_vs_yesterday_pct: number | null;
  status: { level: "GREEN" | "YELLOW" | "RED"; reason: string };
  top_syndromes: { syndrome: string; prob: number; rank: number }[];
  critical_alerts: any[];
  demand_preview: { item_code: string; yhat: number; p10: number; p90: number }[];
  nurse_log_today?: Record<string, any>;
};

function formatLabel(text: string) {
  if (!text) return text;
  return text.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function statusColor(level?: "GREEN" | "YELLOW" | "RED") {
  if (level === "GREEN") return C.green;
  if (level === "YELLOW") return C.yellow;
  if (level === "RED") return C.red;
  return C.sub;
}

export default function Home() {
  const [API_BASE, setAPIBase] = useState(API_DEFAULT);
  const [data, setData] = useState<MobileToday | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // nurse inputs
  const [fever, setFever] = useState("");
  const [cold, setCold] = useState("");
  const [cough, setCough] = useState("");
  const [vomiting, setVomiting] = useState("");
  const [diarrhea, setDiarrhea] = useState("");
  const [others, setOthers] = useState("");
  const [notes, setNotes] = useState("");

  // animated expected patients
  const expectedAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    (async () => {
      const base = await getApiBase(API_DEFAULT);
      setAPIBase(base);
    })();
  }, []);

  const load = useCallback(async () => {
    try {
      setErr(null);
      setLoading(true);
      const d = await apiGet<MobileToday>(API_BASE, "/mobile/today");
      setData(d);
    } catch (e: any) {
      setErr(e?.message ?? "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    load();
  }, [load]);

  // animate when expected changes
  useEffect(() => {
    const target = data?.expected_patients ?? 0;
    Animated.timing(expectedAnim, {
      toValue: target,
      duration: 600,
      useNativeDriver: false,
    }).start();
  }, [data?.expected_patients]);

  const submitLog = async () => {
    try {
      // basic validation: ensure at least one number or notes
      if (!fever && !cough && !cold && !vomiting && !diarrhea && !others && !notes) {
        Alert.alert("SmartCare", "Please enter at least one value or notes before saving.");
        return;
      }

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
      });

      // clear inputs locally after save
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
    const up = data.delta_vs_yesterday_pct >= 0;
    const arrow = up ? "🔺" : "🔻";
    return `${arrow} ${Math.abs(data.delta_vs_yesterday_pct).toFixed(1)}% vs yesterday`;
  }, [data?.delta_vs_yesterday_pct]);

const getGlobalDemandMax = (items: { item_code: string; yhat: number; p10: number; p90: number }[] | undefined) => {
  if (!items || items.length === 0) return 1;
  // use yhat as primary measure; if yhat is missing, fall back to p90
  const vals = items.map((it) => Math.max(0, Number(it.yhat ?? it.p90 ?? 0)));
  const m = Math.max(...vals);
  return m > 0 ? m : 1;
};

const renderDemandPreview = (d: { item_code: string; yhat: number; p10: number; p90: number }, globalMax: number) => {
  // clamp percentage between 0 and 1
  const pct = Math.max(0, Math.min(1, Number(d.yhat || 0) / globalMax));

  return (
    <View key={d.item_code} style={styles.demandRow}>
      <Text style={styles.demandName}>
        {({ antibiotics: "Antibiotics", malaria_kits: "Malaria Kits", ors_packets: "ORS Packets" } as Record<string, string>)[
          d.item_code
        ] || formatLabel(d.item_code)}
      </Text>

      <View style={{ flex: 1, marginLeft: 10 }}>
        {/* global-scaled bar */}
        <View style={styles.rangeBar}>
          <View style={[styles.rangeInner, { width: `${pct * 100}%`, backgroundColor: C.primary }]} />
        </View>

        {/* number shown at the end */}
        <View style={{ flexDirection: "row", justifyContent: "flex-end", marginTop: 6 }}>
          <Text style={{ color: C.text, fontWeight: "700", fontSize: 13 }}>{Math.round(d.yhat)}</Text>
        </View>
      </View>
    </View>
  );
};


  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <ScrollView
        contentContainerStyle={{ padding: 16 }}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={C.primary} />}
      >
        {/* Header */}
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.headerTitle}>Today</Text>
            <Text style={styles.headerSub}>{data?.for_date ?? new Date().toISOString().slice(0, 10)}</Text>
          </View>

          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <TouchableOpacity onPress={load} activeOpacity={0.9} style={styles.iconBtn}>
              <Ionicons name="refresh-outline" size={20} color={C.primary} />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => Alert.alert("API", "Long-press the ⚙️ API label in top-right of the app to change API")}
              activeOpacity={0.9}
              style={styles.iconBtn}
            >
              <Ionicons name="ellipsis-vertical" size={20} color={C.sub} />
            </TouchableOpacity>
          </View>
        </View>

        {/* Expected Patients - hero card */}
        <Card style={styles.heroCard}>
          <View style={{ width: "100%", flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
            <View>
              <Text style={styles.heroLabel}>Expected Patients</Text>

              <Animated.Text style={styles.heroNumber}>
                {expectedAnim.interpolate
                  ? expectedAnim.interpolate({
                      inputRange: [0, 9999],
                      outputRange: [0, 9999],
                    })
                    ? // fallback in case __getValue is not available at runtime for RN Animated.Text interpolation
                      `${data?.expected_patients ?? "—"}`
                    : `${data?.expected_patients ?? "—"}`
                  : `${data?.expected_patients ?? "—"}`}
              </Animated.Text>

              <View style={{ flexDirection: "row", alignItems: "center", marginTop: 8 }}>
                <View style={[styles.deltaPill, { backgroundColor: data?.delta_vs_yesterday_pct && data.delta_vs_yesterday_pct >= 0 ? "#ecfdf5" : "#fff7f7" }]}>
                  <Text style={{ color: data?.delta_vs_yesterday_pct && data.delta_vs_yesterday_pct >= 0 ? C.green : C.red, fontWeight: "700" }}>
                    {deltaText}
                  </Text>
                </View>

                <Text style={{ color: C.sub, marginLeft: 12, fontSize: 12 }}>
                  For {data?.for_date ?? "—"}
                </Text>
              </View>
            </View>

            {/* subtle icon */}
            <View style={styles.heroIconWrap}>
              <Ionicons name="people-outline" size={36} color={C.primary} />
            </View>
          </View>
        </Card>

        {/* Status */}
        <Card accentLeft={statusColor(data?.status?.level)}>
          <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
            <Text style={{ color: C.text, fontWeight: "800" }}>Status</Text>
            {!!data?.status?.level && <Pill text={data.status.level} bg={statusColor(data.status.level)} />}
          </View>
          <Text style={{ color: C.sub, marginTop: 6 }}>{data?.status?.reason ?? ""}</Text>
        </Card>

        {/* Today's Focus */}
        <Card title="Today's Focus">
          <View style={{ flexDirection: "row", flexWrap: "wrap" }}>
            {(data?.top_syndromes || []).length === 0 ? (
              <Text style={{ color: C.sub }}>No dominant syndromes today</Text>
            ) : (
              (data?.top_syndromes || []).map((s) => <Chip key={s.syndrome} label={formatLabel(s.syndrome)} />)
            )}
          </View>
        </Card>

        {/* Critical Alerts preview */}
        <Card title="Critical Alerts">
          {(data?.critical_alerts?.length ?? 0) === 0 ? (
            <Text style={{ color: C.sub }}>No alerts 🎉</Text>
          ) : (
            (data?.critical_alerts || []).map((a, i) => (
              <View key={i} style={styles.alertRow}>
                <View style={{ flex: 1 }}>
                  <Text style={{ color: C.text }}>{String(a.message).replace(/_/g, " ")}</Text>
                </View>
                <Text style={{ color: C.red, fontWeight: "700" }}>{a.severity}</Text>
              </View>
            ))
          )}
        </Card>

        <Card title="Demand Preview">
  {(data?.demand_preview || []).length === 0 ? (
    <Text style={{ color: C.sub }}>No demand preview available</Text>
  ) : (
    (() => {
      const globalMax = getGlobalDemandMax(data?.demand_preview);
      return (data!.demand_preview || []).map((d) => renderDemandPreview(d, globalMax));
    })()
  )}
</Card>


        {/* Log Symptoms - grouped inputs */}
        <Card title="Log Symptoms">
          <View style={styles.rowInputs}>
            <View style={styles.smallInputWrap}>
              <Text style={styles.smallLabel}>Fever</Text>
              <TextInput
                value={fever}
                onChangeText={setFever}
                keyboardType="number-pad"
                placeholder="0"
                placeholderTextColor={C.sub}
                style={styles.smallInput}
              />
            </View>

            <View style={styles.smallInputWrap}>
              <Text style={styles.smallLabel}>Cough</Text>
              <TextInput value={cough} onChangeText={setCough} keyboardType="number-pad" placeholder="0" placeholderTextColor={C.sub} style={styles.smallInput} />
            </View>

            <View style={styles.smallInputWrap}>
              <Text style={styles.smallLabel}>Cold</Text>
              <TextInput value={cold} onChangeText={setCold} keyboardType="number-pad" placeholder="0" placeholderTextColor={C.sub} style={styles.smallInput} />
            </View>
          </View>

          <View style={styles.rowInputs}>
            <View style={styles.smallInputWrap}>
              <Text style={styles.smallLabel}>Diarrhea</Text>
              <TextInput value={diarrhea} onChangeText={setDiarrhea} keyboardType="number-pad" placeholder="0" placeholderTextColor={C.sub} style={styles.smallInput} />
            </View>

            <View style={styles.smallInputWrap}>
              <Text style={styles.smallLabel}>Vomiting</Text>
              <TextInput value={vomiting} onChangeText={setVomiting} keyboardType="number-pad" placeholder="0" placeholderTextColor={C.sub} style={styles.smallInput} />
            </View>

            <View style={styles.smallInputWrap}>
              <Text style={styles.smallLabel}>Others</Text>
              <TextInput value={others} onChangeText={setOthers} keyboardType="number-pad" placeholder="0" placeholderTextColor={C.sub} style={styles.smallInput} />
            </View>
          </View>

          <TextInput
            value={notes}
            onChangeText={setNotes}
            placeholder="Notes (optional)"
            placeholderTextColor={C.sub}
            style={[styles.inputNotes]}
            multiline
            numberOfLines={2}
          />

          <View style={{ marginTop: 12 }}>
            <Button title="Save Log" onPress={submitLog} />
          </View>
        </Card>

        {/* Today's saved log (upgraded UI) */}
<Card title="Today's Log (saved)">
  {data?.nurse_log_today && Object.keys(data.nurse_log_today).length > 0 ? (
    <View>
      {/* header: date + small edit */}
      <View style={styles.logHeader}>
        <View style={{ flexDirection: "row", alignItems: "center" }}>
          <Ionicons name="calendar-outline" size={18} color={C.primary} style={{ marginRight: 8 }} />
          <Text style={styles.logHeaderDate}>{data.nurse_log_today.date}</Text>
        </View>

        <TouchableOpacity
          onPress={() => {
            /* optional: open edit modal / prefill inputs */
            Alert.alert("Edit", "Open edit log flow (not implemented)");
          }}
          activeOpacity={0.9}
          style={styles.editBtn}
        >
          <Ionicons name="pencil-outline" size={16} color={C.primary} />
        </TouchableOpacity>
      </View>

      {/* rows: icon, label, value */}
      <View style={styles.logRows}>
        {data.nurse_log_today.fever != null && (
          <View style={styles.logRowItem}>
            <View style={styles.logRowLeft}>
              <Ionicons name="thermometer-outline" size={18} color={C.red} style={{ marginRight: 12 }} />
              <Text style={styles.logLabel}>Fever</Text>
            </View>
            <Text style={styles.logValue}>{data.nurse_log_today.fever}</Text>
          </View>
        )}

        {data.nurse_log_today.cough != null && (
          <View style={styles.logRowItem}>
            <View style={styles.logRowLeft}>
              <Ionicons name="medical-outline" size={18} color={C.yellow} style={{ marginRight: 12 }} />
              <Text style={styles.logLabel}>Cough</Text>
            </View>
            <Text style={styles.logValue}>{data.nurse_log_today.cough}</Text>
          </View>
        )}

        {data.nurse_log_today.diarrhea != null && (
          <View style={styles.logRowItem}>
            <View style={styles.logRowLeft}>
              <Ionicons name="water-outline" size={18} color="#3b82f6" style={{ marginRight: 12 }} />
              <Text style={styles.logLabel}>Diarrhea</Text>
            </View>
            <Text style={styles.logValue}>{data.nurse_log_today.diarrhea}</Text>
          </View>
        )}

        {data.nurse_log_today.vomiting != null && (
          <View style={styles.logRowItem}>
            <View style={styles.logRowLeft}>
              <Ionicons name="alert-circle-outline" size={18} color={C.red} style={{ marginRight: 12 }} />
              <Text style={styles.logLabel}>Vomiting</Text>
            </View>
            <Text style={styles.logValue}>{data.nurse_log_today.vomiting}</Text>
          </View>
        )}

        {data.nurse_log_today.cold != null && (
          <View style={styles.logRowItem}>
            <View style={styles.logRowLeft}>
              <Ionicons name="snow-outline" size={18} color="#06b6d4" style={{ marginRight: 12 }} />
              <Text style={styles.logLabel}>Cold</Text>
            </View>
            <Text style={styles.logValue}>{data.nurse_log_today.cold}</Text>
          </View>
        )}

        {data.nurse_log_today.others != null && (
          <View style={styles.logRowItem}>
            <View style={styles.logRowLeft}>
              <Ionicons name="medical-outline" size={18} color={C.sub} style={{ marginRight: 12 }} />
              <Text style={styles.logLabel}>Other symptoms</Text>
            </View>
            <Text style={styles.logValue}>{data.nurse_log_today.others}</Text>
          </View>
        )}

        {data.nurse_log_today.notes && (
          <View style={[styles.logRowItem, { alignItems: "flex-start", paddingVertical: 14 }]}>
            <View style={{ flexDirection: "row", alignItems: "center", flex: 1 }}>
              <Ionicons name="document-text-outline" size={18} color={C.sub} style={{ marginRight: 12 }} />
              <View style={{ flex: 1 }}>
                <Text style={styles.logLabel}>Notes</Text>
                <Text style={styles.noteText}>{data.nurse_log_today.notes}</Text>
              </View>
            </View>
          </View>
        )}
      </View>

      {/* footer: who logged it */}
      {data.nurse_log_today.by && (
        <View style={styles.logFooter}>
          <Text style={{ color: C.sub }}>Logged by</Text>
          <Text style={{ color: C.text, fontWeight: "700", marginLeft: 8 }}>{data.nurse_log_today.by}</Text>
        </View>
      )}
    </View>
  ) : (
    <View style={styles.emptyLog}>
      <Ionicons name="document-text-outline" size={42} color={C.sub} style={{ marginBottom: 8 }} />
      <Text style={{ color: C.text, fontWeight: "700", marginBottom: 4 }}>No log for today</Text>
      <Text style={{ color: C.sub }}>Use the Log Symptoms card above to save today's counts</Text>
    </View>
  )}
</Card>


        <View style={{ height: 80 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  headerRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  headerTitle: { color: C.text, fontSize: 22, fontWeight: "800" },
  headerSub: { color: C.sub, fontSize: 13, marginTop: 2 },

  iconBtn: { padding: 8, marginLeft: 8, borderRadius: 10 },

  heroCard: { padding: 18, borderRadius: 16, overflow: "hidden" },
  heroLabel: { color: C.sub, fontSize: 14, fontWeight: "700", marginBottom: 6 },
  heroNumber: { color: C.text, fontSize: 48, fontWeight: "900", marginTop: 2 },
  heroIconWrap: { backgroundColor: "#f0f9ff", padding: 12, borderRadius: 12 },

  deltaPill: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 999 },

  alertRow: { backgroundColor: "#fff7f7", padding: 10, borderRadius: 10, marginBottom: 8 },

  demandRow: { flexDirection: "row", alignItems: "center", marginBottom: 12 },
  demandName: { color: C.text, fontWeight: "700", width: 120 },
  rangeBar: { height: 10, backgroundColor: "#f1f5f9", borderRadius: 6, overflow: "hidden", position: "relative" },
  rangeInner: { position: "absolute", left: 0, top: 0, bottom: 0, borderRadius: 6 },
  rangeInnerTop: { position: "absolute", top: 0, bottom: 0, borderRadius: 6, backgroundColor: "#fdf7e6" },
  rangeMarker: { position: "absolute", top: -4, width: 6, height: 18, borderRadius: 3 },

  rowInputs: { flexDirection: "row", justifyContent: "space-between", marginTop: 8 },
  smallInputWrap: { flex: 1, marginRight: 8 },
  smallLabel: { color: C.sub, fontSize: 12, marginBottom: 6 },
  smallInput: {
    backgroundColor: "#f3f4f6",
    color: C.text,
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: C.border,
    textAlign: "center",
  },

  inputNotes: {
    marginTop: 10,
    backgroundColor: "#f3f4f6",
    color: C.text,
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: C.border,
    minHeight: 44,
    textAlignVertical: "top",
  },
  logHeader: {
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: 8,
},
logHeaderDate: {
  color: C.text,
  fontWeight: "800",
},
editBtn: {
  padding: 8,
  borderRadius: 8,
  backgroundColor: "transparent",
},

logRows: {
  borderTopWidth: 1,
  borderTopColor: "#f1f5f9",
  marginTop: 6,
},

logRowItem: {
  flexDirection: "row",
  alignItems: "center",
  justifyContent: "space-between",
  paddingVertical: 12,
  borderBottomWidth: 1,
  borderBottomColor: "#f3f4f6",
},

logRowLeft: {
  flexDirection: "row",
  alignItems: "center",
  flex: 1,
},

logLabel: {
  color: C.text,
  fontWeight: "700",
},

logValue: {
  color: C.text,
  fontWeight: "900",
  minWidth: 48,
  textAlign: "right",
},

noteText: {
  color: C.text,
  marginTop: 6,
  lineHeight: 18,
},

logFooter: {
  flexDirection: "row",
  alignItems: "center",
  marginTop: 12,
  borderTopWidth: 1,
  borderTopColor: "#f1f5f9",
  paddingTop: 10,
},

emptyLog: {
  alignItems: "center",
  paddingVertical: 18,
},


  logRow: { flexDirection: "row", alignItems: "center", marginTop: 8 },
});
