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
  StyleSheet,
  Alert,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Pill, Chip, ErrorBanner, Button, EmptyState, LoadingSkeleton, Card, Divider, MiniSparkBar } from "../ui";
import { C, getApiBase, API_DEFAULT, formatLabel } from "../constants";
import { apiGet, AlertT, OutbreakAlert, OutbreakResult } from "../api";

const SEVERITY_ORDER = { HIGH: 0, MEDIUM: 1, LOW: 2 };

export default function AlertsScreen({ navigation }: any) {
  const [apiBase, setApiBase] = useState(API_DEFAULT);
  const [alerts, setAlerts] = useState<AlertT[]>([]);
  const [outbreakAlerts, setOutbreakAlerts] = useState<OutbreakAlert[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState<"ALL" | "HIGH" | "MEDIUM" | "LOW">("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);
  const [showOutbreak, setShowOutbreak] = useState(true);

  useEffect(() => {
    (async () => setApiBase(await getApiBase(API_DEFAULT)))();
  }, []);

  const loadAlerts = useCallback(async (isRefresh = false) => {
    try {
      setErr(null);
      if (!isRefresh) setLoading(true);
      const [res, outbreakRes] = await Promise.all([
        apiGet<{ alerts: AlertT[] }>(apiBase, "/alerts"),
        apiGet<OutbreakResult>(apiBase, "/alerts/outbreak").catch(() => null),
      ]);
      setAlerts(res.alerts || []);
      setOutbreakAlerts(outbreakRes?.outbreak_alerts || []);
      setLastFetched(new Date());
    } catch (e: any) {
      setErr(e?.message ?? "Failed to fetch active alerts");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [apiBase]);

  useEffect(() => {
    loadAlerts();
  }, [loadAlerts]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadAlerts(true);
  }, [loadAlerts]);

  const visibleAlerts = useMemo(() => {
    return alerts.filter((_, idx) => !dismissed.has(String(idx)));
  }, [alerts, dismissed]);

  const counts = useMemo(() => ({
    ALL: visibleAlerts.length,
    HIGH: visibleAlerts.filter((a) => a.severity === "HIGH").length,
    MEDIUM: visibleAlerts.filter((a) => a.severity === "MEDIUM").length,
    LOW: visibleAlerts.filter((a) => a.severity === "LOW").length,
  }), [visibleAlerts]);

  const filteredAlerts = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return visibleAlerts
      .filter((a) => filter === "ALL" || a.severity === filter)
      .filter((a) => {
        if (!q) return true;
        return a.item_code.toLowerCase().includes(q) || a.message.toLowerCase().includes(q);
      })
      .sort((a, b) => (SEVERITY_ORDER[a.severity] ?? 3) - (SEVERITY_ORDER[b.severity] ?? 3));
  }, [visibleAlerts, filter, searchQuery]);

  function getSeverityTheme(s: AlertT["severity"]) {
    if (s === "HIGH")   return { color: C.red,    bg: C.redBg,    border: "#fecaca", icon: "alert-circle"    as const };
    if (s === "MEDIUM") return { color: C.yellow,  bg: C.yellowBg, border: "#fde68a", icon: "warning"         as const };
    return                     { color: C.green,   bg: C.greenBg,  border: "#bbf7d0", icon: "information-circle" as const };
  }

  function dismissAlert(originalIndex: number) {
    setDismissed((prev) => {
      const next = new Set(prev);
      next.add(String(originalIndex));
      return next;
    });
  }

  function clearAll() {
    Alert.alert(
      "Clear All Alerts",
      "This will dismiss all currently visible alerts from view. They will reload on next refresh.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Clear All",
          style: "destructive",
          onPress: () => {
            const allIdxs = new Set(alerts.map((_, i) => String(i)));
            setDismissed(allIdxs);
          },
        },
      ]
    );
  }

  function getTimeSince(date: Date | null) {
    if (!date) return null;
    const diff = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  }

  if (loading) {
    return (
      <SafeAreaView style={styles.safe}>
        <LoadingSkeleton />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.primary} />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>Clinical Alerts</Text>
            <Text style={styles.subTitle}>
              Stockout risks & reorder triggers
              {lastFetched && <Text style={{ color: C.textMuted }}> • {getTimeSince(lastFetched)}</Text>}
            </Text>
          </View>

          <View style={{ flexDirection: "row", gap: 8 }}>
            {visibleAlerts.length > 0 && (
              <TouchableOpacity onPress={clearAll} activeOpacity={0.8} style={styles.clearBtn}>
                <Text style={styles.clearBtnText}>Clear All</Text>
              </TouchableOpacity>
            )}
            <View style={styles.counterBadge}>
              <Text style={styles.counterText}>{counts.ALL}</Text>
            </View>
          </View>
        </View>

        <ErrorBanner msg={err} />

        {/* ── Outbreak Surveillance Section ──────── */}
        {outbreakAlerts.length > 0 && showOutbreak && (
          <View style={styles.outbreakSection}>
            <View style={styles.outbreakHeader}>
              <View style={{ flexDirection: "row", alignItems: "center" }}>
                <Ionicons name="cellular" size={16} color={C.red} style={{ marginRight: 6 }} />
                <Text style={styles.outbreakHeaderTitle}>🦠 Syndrome Outbreak Monitor</Text>
              </View>
              <TouchableOpacity onPress={() => setShowOutbreak(false)}>
                <Ionicons name="close" size={18} color={C.sub} />
              </TouchableOpacity>
            </View>
            {outbreakAlerts.map((oa, idx) => (
              <View key={oa.syndrome}>
                <View style={styles.outbreakRow}>
                  <View style={{ flex: 1 }}>
                    <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 4 }}>
                      <Text style={styles.outbreakSyndrome}>{oa.label}</Text>
                      <View style={[
                        styles.ratioBadge,
                        { backgroundColor: oa.severity === "HIGH" ? C.redBg : C.yellowBg }
                      ]}>
                        <Text style={{ color: oa.severity === "HIGH" ? C.red : C.yellow, fontWeight: "900", fontSize: 11 }}>
                          {oa.ratio ? `${oa.ratio}×` : "New"}
                        </Text>
                      </View>
                      <Pill text={oa.severity} bg={oa.severity === "HIGH" ? C.redBg : C.yellowBg}
                        textColor={oa.severity === "HIGH" ? C.red : C.yellow} size="sm" />
                    </View>
                    <Text style={styles.outbreakMsg}>{oa.message}</Text>
                    <View style={{ flexDirection: "row", gap: 12, marginTop: 4, marginBottom: 6 }}>
                      <Text style={styles.outbreakStat}>
                        This week: <Text style={{ color: C.text, fontWeight: "800" }}>{oa.current_7d_cases}</Text>
                      </Text>
                      <Text style={styles.outbreakStat}>
                        Prior: <Text style={{ color: C.sub }}>{oa.baseline_7d_cases}</Text>
                      </Text>
                      <Text style={styles.outbreakStat}>
                        Active {oa.days_with_cases}d
                      </Text>
                    </View>
                    {oa.daily_trend.length > 0 && (
                      <MiniSparkBar
                        data={oa.daily_trend}
                        color={oa.severity === "HIGH" ? C.red : C.yellow}
                        height={22}
                      />
                    )}
                    <Text style={styles.outbreakRec}>{oa.recommendation}</Text>
                  </View>
                </View>
                {idx < outbreakAlerts.length - 1 && <Divider />}
              </View>
            ))}
          </View>
        )}
        {outbreakAlerts.length > 0 && !showOutbreak && (
          <TouchableOpacity onPress={() => setShowOutbreak(true)} style={styles.showOutbreakBtn}>
            <Ionicons name="cellular" size={14} color={C.red} style={{ marginRight: 6 }} />
            <Text style={{ color: C.red, fontWeight: "800", fontSize: 12 }}>
              {outbreakAlerts.length} Outbreak Alert{outbreakAlerts.length > 1 ? "s" : ""} hidden — tap to show
            </Text>
          </TouchableOpacity>
        )}

        {/* Summary row */}
        {visibleAlerts.length > 0 && (
          <View style={styles.summaryRow}>
            {counts.HIGH > 0 && (
              <View style={[styles.summaryChip, { backgroundColor: C.redBg, borderColor: "#fecaca" }]}>
                <Ionicons name="alert-circle" size={14} color={C.red} style={{ marginRight: 4 }} />
                <Text style={[styles.summaryChipText, { color: C.red }]}>{counts.HIGH} HIGH</Text>
              </View>
            )}
            {counts.MEDIUM > 0 && (
              <View style={[styles.summaryChip, { backgroundColor: C.yellowBg, borderColor: "#fde68a" }]}>
                <Ionicons name="warning" size={14} color={C.yellow} style={{ marginRight: 4 }} />
                <Text style={[styles.summaryChipText, { color: C.yellow }]}>{counts.MEDIUM} MEDIUM</Text>
              </View>
            )}
            {counts.LOW > 0 && (
              <View style={[styles.summaryChip, { backgroundColor: C.greenBg, borderColor: "#bbf7d0" }]}>
                <Ionicons name="information-circle" size={14} color={C.green} style={{ marginRight: 4 }} />
                <Text style={[styles.summaryChipText, { color: C.green }]}>{counts.LOW} LOW</Text>
              </View>
            )}
          </View>
        )}

        {/* Filter Chips */}
        <View style={styles.filterRow}>
          {(["ALL", "HIGH", "MEDIUM", "LOW"] as const).map((f) => (
            <Chip key={f} label={f} active={filter === f} count={counts[f]} onPress={() => setFilter(f)} />
          ))}
        </View>

        {/* Search Bar */}
        <View style={styles.searchBox}>
          <Ionicons name="search" size={18} color={C.sub} style={{ marginRight: 8 }} />
          <TextInput
            value={searchQuery}
            onChangeText={setSearchQuery}
            placeholder="Search by drug name or description..."
            placeholderTextColor={C.textMuted}
            style={styles.searchInput}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery("")}>
              <Ionicons name="close-circle" size={18} color={C.sub} />
            </TouchableOpacity>
          )}
        </View>

        {/* Alert Feed */}
        {filteredAlerts.length === 0 ? (
          <EmptyState
            icon="checkmark-done-circle"
            title="All Clear"
            subtitle={
              searchQuery
                ? "No alerts match your search query."
                : dismissed.size > 0
                  ? "All alerts dismissed. Pull to refresh for updated status."
                  : "No stockout or reorder alerts detected. Inventory looks healthy!"
            }
            iconColor={C.green}
            iconBg={C.greenBg}
            action={
              dismissed.size > 0
                ? <Button title="Reload Alerts" variant="outline" onPress={() => { setDismissed(new Set()); onRefresh(); }} icon="refresh-outline" size="sm" />
                : undefined
            }
          />
        ) : (
          filteredAlerts.map((alert, renderIdx) => {
            const originalIdx = alerts.indexOf(alert);
            const theme = getSeverityTheme(alert.severity);

            return (
              <View
                key={`${alert.item_code}-${renderIdx}`}
                style={[styles.alertCard, { borderLeftColor: theme.color }]}
              >
                {/* Top row */}
                <View style={styles.cardHeader}>
                  <View style={{ flexDirection: "row", alignItems: "center", flex: 1 }}>
                    <View style={[styles.alertIconWrap, { backgroundColor: theme.bg }]}>
                      <Ionicons name={theme.icon} size={18} color={theme.color} />
                    </View>
                    <View style={{ marginLeft: 10, flex: 1 }}>
                      <Text style={[styles.typeText, { color: theme.color }]}>
                        {alert.type === "stockout_risk" ? "⚠ STOCKOUT RISK" : "↩ REORDER NEEDED"}
                      </Text>
                      <Text style={styles.itemCodeText}>{formatLabel(alert.item_code)}</Text>
                    </View>
                  </View>
                  <Pill text={alert.severity} bg={theme.bg} textColor={theme.color} dot />
                </View>

                <Divider />

                {/* Message */}
                <Text style={styles.alertMsg}>{alert.message}</Text>

                {/* Actions */}
                <View style={styles.cardActions}>
                  <TouchableOpacity
                    onPress={() => navigation?.navigate("Inventory", { focus: alert.item_code })}
                    activeOpacity={0.8}
                    style={[styles.actionBtn, { backgroundColor: C.primaryBg }]}
                  >
                    <Ionicons name="cube-outline" size={15} color={C.primary} style={{ marginRight: 6 }} />
                    <Text style={[styles.actionBtnText, { color: C.primary }]}>Manage Stock</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    onPress={() => dismissAlert(originalIdx)}
                    activeOpacity={0.8}
                    style={[styles.actionBtn, { backgroundColor: "#f1f5f9" }]}
                  >
                    <Ionicons name="close" size={15} color={C.sub} style={{ marginRight: 4 }} />
                    <Text style={[styles.actionBtnText, { color: C.sub }]}>Dismiss</Text>
                  </TouchableOpacity>
                </View>
              </View>
            );
          })
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  scroll: { padding: 16, paddingBottom: 50 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  title: { fontSize: 22, fontWeight: "900", color: C.text, letterSpacing: -0.3 },
  subTitle: { fontSize: 12, color: C.sub, marginTop: 2 },
  counterBadge: {
    backgroundColor: C.primaryBg, paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: 12, borderWidth: 1, borderColor: "#bae6fd",
    minWidth: 36, alignItems: "center",
  },
  counterText: { color: C.primary, fontWeight: "900", fontSize: 13 },
  clearBtn: {
    backgroundColor: C.redBg, paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: 12, borderWidth: 1, borderColor: "#fecaca",
  },
  clearBtnText: { color: C.red, fontWeight: "800", fontSize: 12 },
  summaryRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginBottom: 10 },
  summaryChip: {
    flexDirection: "row", alignItems: "center",
    paddingVertical: 4, paddingHorizontal: 10, borderRadius: 10, borderWidth: 1,
  },
  summaryChipText: { fontSize: 11, fontWeight: "800" },
  filterRow: { flexDirection: "row", flexWrap: "wrap", marginBottom: 8 },
  searchBox: {
    flexDirection: "row", alignItems: "center", backgroundColor: "#ffffff",
    borderRadius: 14, paddingHorizontal: 14, paddingVertical: 10, marginBottom: 14,
    borderWidth: 1, borderColor: C.inputBorder,
    shadowColor: "#000", shadowOpacity: 0.04, shadowRadius: 4, elevation: 1,
  },
  searchInput: { flex: 1, color: C.text, fontSize: 14 },
  alertCard: {
    backgroundColor: "#ffffff", borderRadius: 18, padding: 16, marginBottom: 12,
    borderWidth: 1, borderColor: C.border, borderLeftWidth: 5,
    shadowColor: "#000", shadowOpacity: 0.06, shadowRadius: 8, elevation: 2,
  },
  cardHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  alertIconWrap: { width: 38, height: 38, borderRadius: 12, alignItems: "center", justifyContent: "center" },
  typeText: { fontSize: 11, fontWeight: "800", letterSpacing: 0.3 },
  itemCodeText: { color: C.text, fontSize: 14, fontWeight: "800", marginTop: 2 },
  alertMsg: { color: C.textSecondary, fontSize: 14, fontWeight: "600", lineHeight: 20, marginVertical: 6 },
  cardActions: { flexDirection: "row", gap: 10, marginTop: 4 },
  actionBtn: {
    flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center",
    paddingVertical: 8, paddingHorizontal: 12, borderRadius: 10,
  },
  actionBtnText: { fontSize: 13, fontWeight: "800" },
  // Outbreak section
  outbreakSection: {
    backgroundColor: "#fff7ed", borderRadius: 18, padding: 14, marginBottom: 12,
    borderWidth: 1, borderColor: "#fed7aa",
  },
  outbreakHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  outbreakHeaderTitle: { color: C.red, fontWeight: "800", fontSize: 14 },
  outbreakRow: { paddingVertical: 8 },
  outbreakSyndrome: { color: C.text, fontWeight: "800", fontSize: 14, marginRight: 8 },
  ratioBadge: { paddingHorizontal: 7, paddingVertical: 2, borderRadius: 7, marginRight: 6 },
  outbreakMsg: { color: C.textSecondary, fontSize: 12, fontWeight: "600", marginBottom: 2 },
  outbreakStat: { color: C.sub, fontSize: 11, fontWeight: "600" },
  outbreakRec: { color: C.sub, fontSize: 11, fontStyle: "italic", marginTop: 6 },
  showOutbreakBtn: {
    flexDirection: "row", alignItems: "center", justifyContent: "center",
    backgroundColor: C.redBg, borderRadius: 12, paddingVertical: 10, marginBottom: 10,
    borderWidth: 1, borderColor: "#fecaca",
  },
});

