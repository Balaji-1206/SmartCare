// src/pages/Alerts.tsx
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  SafeAreaView,
  View,
  Text,
  ScrollView,
  RefreshControl,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
  Alert,
  Pressable,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Card, Pill } from "../ui";
import { C, getApiBase, API_DEFAULT } from "../constants";
import { apiGet } from "../api";

type AlertT = {
  type: "stockout_risk" | "reorder";
  severity: "HIGH" | "MEDIUM" | "LOW";
  message: string;
  item_code: string;
};

export default function AlertsScreen({ navigation }: any) {
  const [api, setApi] = useState(API_DEFAULT);
  const [alerts, setAlerts] = useState<AlertT[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<"ALL" | "HIGH" | "MEDIUM" | "LOW">("ALL");
  const [query, setQuery] = useState("");
  const [ackQueue, setAckQueue] = useState<Record<string, NodeJS.Timeout>>({}); // for undo toasts
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  useEffect(() => {
    (async () => setApi(await getApiBase(API_DEFAULT)))();
  }, []);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      setStatusMsg(null);
      const j = await apiGet<{ alerts: AlertT[] }>(api, "/alerts");
      setAlerts(j.alerts || []);
    } catch (e: any) {
      const m = e?.message ?? "Failed to load alerts";
      setStatusMsg(m);
      Alert.alert("SmartCare", m);
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    load();
  }, [load]);

  // derived list
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (filter === "ALL" ? alerts : alerts.filter((a) => a.severity === filter)).filter((a) => {
      if (!q) return true;
      return a.item_code.toLowerCase().includes(q) || a.message.toLowerCase().includes(q);
    });
  }, [alerts, filter, query]);

  function severityColor(s: AlertT["severity"]) {
    if (s === "HIGH") return C.red;
    if (s === "MEDIUM") return C.yellow;
    return C.green;
  }

  async function openInventory(code: string) {
    // navigate to Inventory and focus item
    navigation.navigate("Inventory", { focus: code });
  }

  // quick acknowledge (local only) with undo
  function acknowledge(key: string) {
    // remove from list immediately (optimistic)
    setAlerts((prev) => prev.filter((a, i) => `${a.item_code}-${i}` !== key));
    setStatusMsg("Acknowledged — undo available");
    const t = setTimeout(() => {
      // permanent (in this simple implementation we just remove locally)
      setStatusMsg(null);
      setAckQueue((q) => {
        const copy = { ...q };
        delete copy[key];
        return copy;
      });
    }, 4000);
    setAckQueue((q) => ({ ...q, [key]: t }));
  }

  function undoAcknowledge(key: string) {
    // cancel undo timer and reload from server to restore state
    const t = ackQueue[key];
    if (t) clearTimeout(t);
    setAckQueue((q) => {
      const copy = { ...q };
      delete copy[key];
      return copy;
    });
    setStatusMsg("Restored");
    load(); // reload fresh
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <ScrollView
        contentContainerStyle={{ padding: 16 }}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={C.primary} />}
        keyboardShouldPersistTaps="handled"
      >
        <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 12 }}>
          <Ionicons name="alert-circle-outline" size={24} color={C.red} style={{ marginRight: 8 }} />
          <Text style={{ color: C.text, fontSize: 22, fontWeight: "800" }}>Alerts</Text>
        </View>

        {/* header controls */}
        <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 12 }}>
          <TextInput
            placeholder="Search item or message..."
            placeholderTextColor={C.sub}
            value={query}
            onChangeText={setQuery}
            style={styles.searchInput}
          />

          <View style={{ marginLeft: 10, flexDirection: "row" }}>
            {(["ALL", "HIGH", "MEDIUM", "LOW"] as const).map((f) => {
              const active = filter === f;
              return (
                <TouchableOpacity
                  key={f}
                  onPress={() => setFilter(f)}
                  activeOpacity={0.9}
                  style={[
                    styles.filterBtn,
                    { backgroundColor: active ? C.primary : C.chip, marginLeft: f === "ALL" ? 0 : 8 },
                  ]}
                >
                  <Text style={{ color: active ? "#fff" : C.text, fontWeight: "700", fontSize: 12 }}>{f}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* status message */}
        {!!statusMsg && (
          <View style={{ marginBottom: 10 }}>
            <Text style={{ color: C.sub }}>{statusMsg}</Text>
          </View>
        )}

        {/* empty state */}
        {!loading && shown.length === 0 ? (
          <Card style={{ alignItems: "center", paddingVertical: 40 }}>
            <Ionicons name="checkmark-done-circle-outline" size={48} color={C.green} style={{ marginBottom: 10 }} />
            <Text style={{ color: C.text, fontWeight: "700", marginBottom: 6 }}>No alerts</Text>
            <Text style={{ color: C.sub }}>You're all clear — pull to refresh</Text>
          </Card>
        ) : null}

        {/* list */}
        {loading ? (
          <View>
            <Card style={{ marginBottom: 12 }}>
              <View style={{ height: 60, justifyContent: "center" }}>
                <ActivityIndicator color={C.primary} />
              </View>
            </Card>
          </View>
        ) : (
          shown.map((a, idx) => {
            const key = `${a.item_code}-${idx}`;
            return (
              <Card
                key={key}
                style={{
                  marginBottom: 12,
                  borderLeftWidth: 6,
                  borderLeftColor: severityColor(a.severity),
                }}
              >
                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <View style={{ flex: 1, paddingRight: 8 }}>
                    <Text style={{ color: C.text, fontWeight: "800", marginBottom: 6 }}>
                      {a.type === "stockout_risk" ? "Stock-out risk" : "Reorder"}
                    </Text>
                    <Text style={{ color: C.sub, marginBottom: 8 }}>{a.message}</Text>

                    <View style={{ flexDirection: "row", marginTop: 4, alignItems: "center", flexWrap: "wrap" }}>
                      
                      <Pill text={a.severity} bg={severityColor(a.severity)} />
                    </View>
                  </View>

                  <View style={{ alignItems: "flex-end", justifyContent: "space-between" }}>
                    <TouchableOpacity
                      onPress={() => acknowledge(key)}
                      activeOpacity={0.9}
                      style={{
                        backgroundColor: "#f3f4f6",
                        paddingVertical: 8,
                        paddingHorizontal: 12,
                        borderRadius: 10,
                        marginBottom: 8,
                      }}
                    >
                      <Text style={{ color: C.text, fontWeight: "700" }}>Acknowledge</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
  onPress={() => openInventory(a.item_code)}
  activeOpacity={0.9}
  style={{
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: C.primary,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
    marginTop: 6,
    shadowColor: C.shadow,
    shadowOpacity: 0.15,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  }}
>
  <Ionicons name="cube-outline" size={16} color="#fff" style={{ marginRight: 6 }} />
  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 13 }}>Open Inventory</Text>
</TouchableOpacity>

                  </View>
                </View>

                {/* undo bar */}
                {ackQueue[key] ? (
                  <View style={{ marginTop: 12, flexDirection: "row", justifyContent: "flex-end" }}>
                    <TouchableOpacity
                      onPress={() => undoAcknowledge(key)}
                      activeOpacity={0.9}
                      style={{ paddingHorizontal: 12, paddingVertical: 8 }}
                    >
                      <Text style={{ color: C.primary, fontWeight: "700" }}>Undo</Text>
                    </TouchableOpacity>
                  </View>
                ) : null}
              </Card>
            );
          })
        )}

        <View style={{ height: 80 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  searchInput: {
    flex: 1,
    backgroundColor: "#fff",
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: C.border,
    color: C.text,
  },
  filterBtn: {
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 999,
  },
});
