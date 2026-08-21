// src/pages/Home.tsx
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  SafeAreaView,
  View,
  Text,
  ScrollView,
  RefreshControl,
  TextInput,
  TouchableOpacity,
  Modal,
  Alert,
  StyleSheet,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import {
  Card,
  StatCard,
  SectionTitle,
  Pill,
  Button,
  StepperInput,
  ProgressBar,
  ErrorBanner,
  LoadingSkeleton,
  Divider,
  AnimatedNumber,
  NurseCalendarStrip,
  MiniSparkBar,
} from "../ui";
import { C, getApiBase, API_DEFAULT, getNurseName, formatLabel } from "../constants";
import {
  apiGet,
  MobileToday,
  StatsSummary,
  NurseLogDay,
  OutbreakAlert,
  OutbreakResult,
} from "../api";
import { onlineOrQueue } from "../offlineQueue";

function getStatusTheme(level?: "GREEN" | "YELLOW" | "RED") {
  if (level === "RED")    return { bg: C.redBg,    color: C.red,    label: "Surge Alert",   icon: "trending-up"       as const };
  if (level === "YELLOW") return { bg: C.yellowBg, color: C.yellow, label: "Elevated Load", icon: "warning"           as const };
  return                         { bg: C.greenBg,  color: C.green,  label: "Normal Flow",   icon: "checkmark-circle"  as const };
}

export default function Home({ navigation }: any) {
  const [apiBase, setApiBase] = useState(API_DEFAULT);
  const [nurseName, setNurseName] = useState("");
  const [data, setData] = useState<MobileToday | null>(null);
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [nurseHistory, setNurseHistory] = useState<NurseLogDay[]>([]);
  const [outbreakAlerts, setOutbreakAlerts] = useState<OutbreakAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [savingLog, setSavingLog] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [logSaved, setLogSaved] = useState(false);

  // Nurse Symptom Steppers
  const [fever, setFever] = useState(0);
  const [cough, setCough] = useState(0);
  const [diarrhea, setDiarrhea] = useState(0);
  const [vomiting, setVomiting] = useState(0);
  const [cold, setCold] = useState(0);
  const [others, setOthers] = useState(0);
  const [notes, setNotes] = useState("");

  // Calendar day detail modal
  const [selectedDay, setSelectedDay] = useState<NurseLogDay | null>(null);

  useEffect(() => {
    (async () => {
      const base = await getApiBase(API_DEFAULT);
      setApiBase(base);
      const name = await getNurseName();
      if (name) setNurseName(name);
    })();
  }, []);

  const loadAll = useCallback(async () => {
    try {
      setErr(null);
      const [todayData, statsData, historyData, outbreakData] = await Promise.all([
        apiGet<MobileToday>(apiBase, "/mobile/today").catch(() => null),
        apiGet<StatsSummary>(apiBase, "/stats/summary").catch(() => null),
        apiGet<{ days: number; history: NurseLogDay[] }>(apiBase, "/nurse/log/history?days=7").catch(() => null),
        apiGet<OutbreakResult>(apiBase, "/alerts/outbreak").catch(() => null),
      ]);

      if (todayData) {
        setData(todayData);
        const log = todayData.nurse_log_today;
        if (log && Object.keys(log).length > 0) {
          setFever(log.fever || 0);
          setCough(log.cough || 0);
          setDiarrhea(log.diarrhea || 0);
          setVomiting(log.vomiting || 0);
          setCold(log.cold || 0);
          setOthers(log.others || 0);
          setNotes(log.notes || "");
        }
      }
      if (statsData)   setStats(statsData);
      if (historyData) setNurseHistory(historyData.history || []);
      if (outbreakData) setOutbreakAlerts(outbreakData.outbreak_alerts || []);
    } catch (e: any) {
      setErr(e?.message ?? "Failed to fetch clinic telemetry");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [apiBase]);

  useEffect(() => {
    if (apiBase) loadAll();
  }, [loadAll]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadAll();
  }, [loadAll]);

  const submitLog = async () => {
    const totalSymptoms = fever + cough + diarrhea + vomiting + cold + others;
    if (totalSymptoms === 0 && !notes.trim()) {
      Alert.alert("Empty Log", "Please record at least one symptom count or clinical observation.");
      return;
    }
    try {
      setSavingLog(true);
      const today = new Date().toISOString().slice(0, 10);
      const payload = {
        date: today,
        fever: fever || undefined,
        cough: cough || undefined,
        diarrhea: diarrhea || undefined,
        vomiting: vomiting || undefined,
        cold: cold || undefined,
        others: others || undefined,
        notes: notes.trim() || undefined,
        by: nurseName || "On-duty Nurse",
      };
      const res = await onlineOrQueue(apiBase, "/nurse/log", payload);
      setLogSaved(true);
      setTimeout(() => setLogSaved(false), 3000);
      if (res.queued) {
        Alert.alert("Offline Sync", "Submission queued locally. Auto-syncs when connected.");
      } else {
        await loadAll();
      }
    } catch (e: any) {
      Alert.alert("Submission Error", e?.message ?? "Failed to record log");
    } finally {
      setSavingLog(false);
    }
  };

  const statusTheme = getStatusTheme(data?.status?.level);
  const totalSymptomsToday = useMemo(() => {
    const log = data?.nurse_log_today;
    if (!log) return null;
    return (log.fever || 0) + (log.cough || 0) + (log.diarrhea || 0) + (log.vomiting || 0) + (log.cold || 0) + (log.others || 0);
  }, [data?.nurse_log_today]);

  const deltaText = useMemo(() => {
    if (data?.delta_vs_yesterday_pct == null) return null;
    const val = data.delta_vs_yesterday_pct;
    return { isUp: val > 0, abs: Math.abs(val).toFixed(1), val };
  }, [data?.delta_vs_yesterday_pct]);

  const greeting = useMemo(() => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 17) return "Good afternoon";
    return "Good evening";
  }, []);

  if (loading && !data) {
    return <SafeAreaView style={styles.safe}><LoadingSkeleton /></SafeAreaView>;
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.primary} />}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Header ───────────────────────────────────────── */}
        <View style={styles.topHeader}>
          <View style={{ flex: 1 }}>
            <Text style={styles.greetingText}>
              {greeting}{nurseName ? `, ${nurseName.split(" ")[0]}` : ""}
            </Text>
            <Text style={styles.dateBanner}>
              📅 {data?.for_date || new Date().toISOString().slice(0, 10)} · PHC Daily Overview
            </Text>
          </View>
          <TouchableOpacity onPress={onRefresh} activeOpacity={0.8} style={styles.headerBtn}>
            <Ionicons name="refresh" size={18} color={C.primary} />
          </TouchableOpacity>
        </View>

        <ErrorBanner msg={err} />
        {logSaved && <ErrorBanner msg="Triage log saved successfully!" type="success" />}

        {/* ── Outbreak Alert Banner ────────────────────────── */}
        {outbreakAlerts.length > 0 && (
          <View style={[
            styles.outbreakBanner,
            { borderColor: outbreakAlerts[0].severity === "HIGH" ? "#fecaca" : "#fde68a",
              backgroundColor: outbreakAlerts[0].severity === "HIGH" ? C.redBg : C.yellowBg }
          ]}>
            <View style={styles.outbreakRow}>
              <View style={[styles.outbreakIcon, {
                backgroundColor: outbreakAlerts[0].severity === "HIGH" ? "#fee2e2" : "#fef3c7"
              }]}>
                <Ionicons
                  name="warning"
                  size={20}
                  color={outbreakAlerts[0].severity === "HIGH" ? C.red : C.yellow}
                />
              </View>
              <View style={{ flex: 1, marginLeft: 10 }}>
                <Text style={[styles.outbreakTitle, {
                  color: outbreakAlerts[0].severity === "HIGH" ? C.red : C.yellow
                }]}>
                  🦠 Outbreak Alert — {outbreakAlerts.length} syndrome{outbreakAlerts.length > 1 ? "s" : ""} flagged
                </Text>
                <Text style={styles.outbreakMsg} numberOfLines={2}>
                  {outbreakAlerts[0].message}
                </Text>
              </View>
            </View>

            {outbreakAlerts.map((alert, idx) => (
              <View key={alert.syndrome} style={styles.outbreakItem}>
                <View style={{ flex: 1 }}>
                  <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 4 }}>
                    <Text style={styles.outbreakSyndrome}>{alert.label}</Text>
                    <Pill
                      text={alert.severity}
                      bg={alert.severity === "HIGH" ? "#fee2e2" : "#fef3c7"}
                      textColor={alert.severity === "HIGH" ? C.red : C.yellow}
                      size="sm"
                    />
                    {alert.ratio && (
                      <Text style={{ color: C.red, fontWeight: "900", fontSize: 12, marginLeft: 8 }}>
                        {alert.ratio}× surge
                      </Text>
                    )}
                  </View>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
                    <Text style={{ color: C.textSecondary, fontSize: 12, fontWeight: "600" }}>
                      This week: <Text style={{ color: C.text, fontWeight: "800" }}>{alert.current_7d_cases}</Text>
                    </Text>
                    <Text style={{ color: C.textSecondary, fontSize: 12, fontWeight: "600" }}>
                      Prior: <Text style={{ color: C.sub }}>{alert.baseline_7d_cases}</Text>
                    </Text>
                  </View>
                  {alert.daily_trend.length > 0 && (
                    <MiniSparkBar
                      data={alert.daily_trend}
                      color={alert.severity === "HIGH" ? C.red : C.yellow}
                      height={24}
                    />
                  )}
                  <Text style={{ color: C.sub, fontSize: 11, marginTop: 4, fontStyle: "italic" }}>
                    {alert.recommendation}
                  </Text>
                </View>
                {idx < outbreakAlerts.length - 1 && <Divider />}
              </View>
            ))}
          </View>
        )}

        {/* ── Hero Surge Card with Animated Counter ────────── */}
        <View style={[styles.heroCard, { borderLeftColor: statusTheme.color }]}>
          <View style={styles.heroTopRow}>
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <View style={[styles.statusIconWrap, { backgroundColor: statusTheme.bg }]}>
                <Ionicons name="pulse" size={20} color={statusTheme.color} />
              </View>
              <View style={{ marginLeft: 10 }}>
                <Text style={styles.heroSubHeader}>Patient Volume Forecast</Text>
                <Text style={[styles.statusTitle, { color: statusTheme.color }]}>{statusTheme.label}</Text>
              </View>
            </View>
            <Pill text={data?.status?.level || "—"} bg={statusTheme.bg} textColor={statusTheme.color} dot />
          </View>

          {/* Animated count-up number */}
          <View style={styles.heroNumberSection}>
            <AnimatedNumber
              value={data?.expected_patients != null ? Math.round(data.expected_patients) : 0}
              textStyle={styles.mainNumber}
            />
            <Text style={styles.mainNumberUnit}>expected visits today</Text>
            {data?.expected_patients_p10 != null && data?.expected_patients_p90 != null && (
              <Text style={styles.confidenceText}>
                Confidence range: {Math.round(data.expected_patients_p10)} – {Math.round(data.expected_patients_p90)} patients
              </Text>
            )}
          </View>

          {/* 7-day sparkline from last7_volumes */}
          {data?.last7_volumes && data.last7_volumes.length >= 3 && (
            <View style={{ marginBottom: 12 }}>
              <Text style={{ color: C.sub, fontSize: 11, fontWeight: "600", marginBottom: 6 }}>
                7-Day Visit Trend
              </Text>
              <MiniSparkBar data={data.last7_volumes} color={statusTheme.color} height={32} />
            </View>
          )}

          <View style={styles.heroFooter}>
            {deltaText ? (
              <View style={[styles.deltaBadge, { backgroundColor: deltaText.isUp ? C.yellowBg : C.greenBg }]}>
                <Ionicons
                  name={deltaText.isUp ? "trending-up" : "trending-down"}
                  size={13}
                  color={deltaText.isUp ? C.yellow : C.green}
                  style={{ marginRight: 4 }}
                />
                <Text style={[styles.deltaText, { color: deltaText.isUp ? C.yellow : C.green }]}>
                  {deltaText.isUp ? "+" : "-"}{deltaText.abs}% vs yesterday
                </Text>
              </View>
            ) : <View />}
            <Text style={styles.statusReason}>{data?.status?.reason}</Text>
          </View>
        </View>

        {/* ── Quick Stats Row ───────────────────────────────── */}
        <View style={styles.statsRow}>
          <StatCard
            label="7-Day Avg"
            value={stats?.last7.mean != null ? Math.round(stats.last7.mean) : "—"}
            unit="pts"
            icon="bar-chart-outline"
            trend={stats?.trend_7d_pct}
            style={{ marginRight: 8 }}
          />
          {data?.weather?.temperature != null ? (
            <StatCard
              label="Temperature"
              value={`${data.weather.temperature.toFixed(0)}°C`}
              icon="thermometer-outline"
              iconColor={C.red}
              iconBg={C.redBg}
              style={{ marginLeft: 8 }}
            />
          ) : (
            <StatCard
              label="Humidity"
              value={data?.weather?.humidity != null ? `${data.weather.humidity}%` : "N/A"}
              icon="water-outline"
              style={{ marginLeft: 8 }}
            />
          )}
        </View>

        {/* ── Alert Banner ─────────────────────────────────── */}
        {data?.critical_alerts && data.critical_alerts.length > 0 && (
          <TouchableOpacity onPress={() => navigation?.navigate("Alerts")} activeOpacity={0.9} style={styles.alertBanner}>
            <View style={styles.alertBannerLeft}>
              <View style={styles.alertIconCircle}>
                <Ionicons name="alert-circle" size={22} color={C.red} />
              </View>
              <View style={{ marginLeft: 12, flex: 1 }}>
                <Text style={styles.alertBannerTitle}>
                  {data.all_alerts_count ?? data.critical_alerts.length} Active Stock Alert{(data.all_alerts_count ?? 0) !== 1 ? "s" : ""}
                </Text>
                <Text style={styles.alertBannerSub} numberOfLines={1}>
                  {data.critical_alerts[0].message}
                </Text>
              </View>
            </View>
            <Ionicons name="chevron-forward" size={18} color={C.red} />
          </TouchableOpacity>
        )}

        {/* ── 7-Day Nurse Log Calendar ──────────────────────── */}
        {nurseHistory.length > 0 && (
          <>
            <SectionTitle
              title="7-Day Triage Calendar"
              icon="calendar-outline"
              rightAction={<Text style={styles.sectionMeta}>Tap a day to see details</Text>}
            />
            <NurseCalendarStrip history={nurseHistory} onDayPress={(d) => setSelectedDay(d)} />
          </>
        )}

        {/* ── Syndromic Surveillance ────────────────────────── */}
        <SectionTitle
          title="Syndromic Surveillance"
          icon="shield-checkmark-outline"
          rightAction={<Text style={styles.sectionMeta}>Top Risks Today</Text>}
        />
        <Card style={{ marginTop: 6, padding: 14 }}>
          {(!data?.top_syndromes || data.top_syndromes.length === 0) ? (
            <Text style={{ color: C.sub, textAlign: "center", paddingVertical: 10, fontWeight: "600" }}>
              ✅ No abnormal syndrome risks detected
            </Text>
          ) : (
            data.top_syndromes.map((syn, idx) => {
              const pct = Math.round(syn.prob * 100);
              const isHigh = syn.prob >= 0.75;
              const isMed  = syn.prob >= 0.5;
              const barColor = isHigh ? C.red : isMed ? C.yellow : C.primary;
              const rankBg   = isHigh ? C.redBg : isMed ? C.yellowBg : C.primaryBg;
              const rankCol  = isHigh ? C.red : isMed ? C.yellow : C.primary;
              return (
                <View key={syn.syndrome}>
                  <View style={styles.synRow}>
                    <View style={styles.synLeft}>
                      <View style={[styles.rankPill, { backgroundColor: rankBg }]}>
                        <Text style={[styles.rankText, { color: rankCol }]}>#{idx + 1}</Text>
                      </View>
                      <View style={{ marginLeft: 10 }}>
                        <Text style={styles.synName}>{formatLabel(syn.syndrome)}</Text>
                        <Text style={[styles.synRiskLabel, { color: barColor }]}>
                          {isHigh ? "High Risk" : isMed ? "Moderate" : "Low Risk"}
                        </Text>
                      </View>
                    </View>
                    <View style={styles.synRight}>
                      <Text style={[styles.synPercent, { color: barColor }]}>{pct}%</Text>
                      <ProgressBar progress={syn.prob} color={barColor} height={5} style={{ width: 70, marginTop: 4 }} />
                    </View>
                  </View>
                  {idx < data.top_syndromes.length - 1 && <Divider />}
                </View>
              );
            })
          )}
        </Card>

        {/* ── Medicine Demand Preview ───────────────────────── */}
        <SectionTitle
          title="Medicine Demand Forecast"
          icon="medkit-outline"
          rightAction={
            <TouchableOpacity onPress={() => navigation?.navigate("Inventory")} activeOpacity={0.8}>
              <Text style={styles.linkText}>View Inventory →</Text>
            </TouchableOpacity>
          }
        />
        <View style={styles.demandGrid}>
          {(data?.demand_preview || []).map((item) => (
            <View key={item.item_code} style={styles.demandCard}>
              <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" }}>
                <Text style={styles.demandItemName}>{formatLabel(item.item_code)}</Text>
                <Pill text="Daily" bg={C.cyanBg} textColor={C.cyan} size="sm" />
              </View>
              <AnimatedNumber
                value={Math.round(item.yhat)}
                textStyle={styles.demandUnits}
                duration={900}
              />
              <Text style={styles.demandUnit}>units/day</Text>
              {item.p10 != null && item.p90 != null && (
                <Text style={styles.demandRange}>
                  Range: {Math.round(item.p10)}–{Math.round(item.p90)}
                </Text>
              )}
              <ProgressBar progress={Math.min(1, item.yhat / 15)} color={C.cyan} height={4} style={{ marginTop: 8 }} />
            </View>
          ))}
        </View>

        {/* ── Today's Nurse Log Summary ─────────────────────── */}
        {data?.nurse_log_today && Object.keys(data.nurse_log_today).length > 0 && (
          <>
            <SectionTitle title="Today's Logged Cases" icon="clipboard-outline" />
            <Card accentLeft={C.primary} style={{ marginTop: 6 }}>
              <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <Text style={{ color: C.text, fontWeight: "800", fontSize: 15 }}>
                  Triage Summary · {data.nurse_log_today.date || data.for_date}
                </Text>
                {totalSymptomsToday != null && (
                  <Pill text={`${totalSymptomsToday} total`} bg={C.primaryBg} textColor={C.primary} />
                )}
              </View>
              {[
                { key: "fever",    label: "Fever",    icon: "thermometer-outline",    color: C.red },
                { key: "cough",    label: "Cough",    icon: "fitness-outline",         color: C.yellow },
                { key: "diarrhea", label: "Diarrhea", icon: "water-outline",           color: C.cyan },
                { key: "vomiting", label: "Vomiting", icon: "refresh-circle-outline",  color: C.purple },
                { key: "cold",     label: "Cold",     icon: "snow-outline",            color: C.primary },
                { key: "others",   label: "Others",   icon: "help-circle-outline",     color: C.sub },
              ]
                .filter(({ key }) => (data.nurse_log_today?.[key] || 0) > 0)
                .map(({ key, label, icon, color }) => (
                  <View key={key} style={styles.logRow}>
                    <Ionicons name={icon as any} size={16} color={color} style={{ marginRight: 8 }} />
                    <Text style={styles.logLabel}>{label}</Text>
                    <Text style={[styles.logValue, { color }]}>{data.nurse_log_today![key] || 0}</Text>
                  </View>
                ))}
              {data.nurse_log_today.notes ? (
                <View style={styles.notesSnippet}>
                  <Ionicons name="document-text-outline" size={14} color={C.sub} style={{ marginRight: 6 }} />
                  <Text style={styles.notesSnippetText} numberOfLines={2}>{data.nurse_log_today.notes}</Text>
                </View>
              ) : null}
              {data.nurse_log_today.by ? (
                <Text style={styles.loggedByText}>Logged by: {data.nurse_log_today.by}</Text>
              ) : null}
            </Card>
          </>
        )}

        {/* ── Nurse Daily Triage Logger ─────────────────────── */}
        <SectionTitle title="Daily Nurse Triage Logger" icon="create-outline" />
        <Card
          title="Add / Update Today's Entries"
          subtitle="Record clinical presentations observed at triage"
          icon="clipboard-outline"
        >
          <StepperInput label="Fever & High Temp" sublabel="Febrile illness, malaria suspicion" value={fever} onChange={setFever} icon="thermometer-outline" color={C.red} />
          <StepperInput label="Cough & Respiratory" sublabel="ARI, pneumonia, TB screening" value={cough} onChange={setCough} icon="fitness-outline" color={C.yellow} />
          <StepperInput label="Diarrhea & GI" sublabel="Dehydration, cholera watch" value={diarrhea} onChange={setDiarrhea} icon="water-outline" color={C.cyan} />
          <StepperInput label="Vomiting / Nausea" sublabel="Food poisoning, gastroenteritis" value={vomiting} onChange={setVomiting} icon="refresh-circle-outline" color={C.purple} />
          <StepperInput label="Cold / URTI" sublabel="Rhinitis, seasonal flu" value={cold} onChange={setCold} icon="snow-outline" color={C.primary} />
          <StepperInput label="Others / Unclassified" sublabel="Skin rash, animal bite, etc." value={others} onChange={setOthers} icon="help-circle-outline" color={C.sub} />

          <View style={styles.notesBox}>
            <Text style={styles.notesLabel}>Clinical Observations</Text>
            <TextInput
              value={notes}
              onChangeText={setNotes}
              placeholder="e.g. Cluster of fever cases from Sector 3 school..."
              placeholderTextColor={C.textMuted}
              multiline
              numberOfLines={4}
              style={styles.notesInput}
            />
          </View>

          <View style={{ marginTop: 14, gap: 10 }}>
            <Button
              title={savingLog ? "Saving Triage Log..." : "Save Today's Triage Entries"}
              onPress={submitLog}
              loading={savingLog}
              icon="cloud-upload-outline"
            />
            {(fever + cough + diarrhea + vomiting + cold + others > 0 || notes.trim()) && (
              <Button
                title="Clear Form"
                variant="ghost"
                size="sm"
                onPress={() => {
                  setFever(0); setCough(0); setDiarrhea(0);
                  setVomiting(0); setCold(0); setOthers(0); setNotes("");
                }}
                icon="close-circle-outline"
              />
            )}
          </View>
        </Card>
      </ScrollView>

      {/* ── Calendar Day Detail Modal ─────────────────────── */}
      <Modal visible={!!selectedDay} transparent animationType="fade" onRequestClose={() => setSelectedDay(null)}>
        <TouchableOpacity
          style={styles.modalBackdrop}
          activeOpacity={1}
          onPress={() => setSelectedDay(null)}
        >
          <View style={styles.dayModal} onStartShouldSetResponder={() => true}>
            <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
              <Text style={styles.dayModalTitle}>
                {selectedDay?.date || "—"}
              </Text>
              <TouchableOpacity onPress={() => setSelectedDay(null)}>
                <Ionicons name="close" size={22} color={C.sub} />
              </TouchableOpacity>
            </View>
            <Divider />
            {selectedDay && !selectedDay.has_entry ? (
              <Text style={{ color: C.sub, textAlign: "center", paddingVertical: 20, fontWeight: "600" }}>
                No triage entries logged for this day.
              </Text>
            ) : selectedDay ? (
              <>
                {[
                  { key: "fever",    label: "Fever",    color: C.red    },
                  { key: "cough",    label: "Cough",    color: C.yellow },
                  { key: "diarrhea", label: "Diarrhea", color: C.cyan   },
                  { key: "vomiting", label: "Vomiting", color: C.purple },
                  { key: "cold",     label: "Cold",     color: C.primary},
                  { key: "others",   label: "Others",   color: C.sub    },
                ]
                  .filter(({ key }) => (selectedDay as any)[key] > 0)
                  .map(({ key, label, color }) => (
                    <View key={key} style={styles.logRow}>
                      <Text style={styles.logLabel}>{label}</Text>
                      <Text style={[styles.logValue, { color }]}>{(selectedDay as any)[key]}</Text>
                    </View>
                  ))}
                <View style={[styles.logRow, { justifyContent: "space-between", marginTop: 8, paddingTop: 8, borderTopWidth: 1, borderTopColor: C.divider }]}>
                  <Text style={{ color: C.textSecondary, fontWeight: "700" }}>Total Cases</Text>
                  <Text style={{ color: C.text, fontWeight: "900", fontSize: 16 }}>
                    {selectedDay.fever + selectedDay.cough + selectedDay.diarrhea +
                     selectedDay.vomiting + selectedDay.cold + selectedDay.others}
                  </Text>
                </View>
              </>
            ) : null}
          </View>
        </TouchableOpacity>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  scroll: { padding: 16, paddingBottom: 50 },
  topHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  greetingText: { fontSize: 20, fontWeight: "900", color: C.text, letterSpacing: -0.4 },
  dateBanner: { fontSize: 12, color: C.sub, marginTop: 3, fontWeight: "600" },
  headerBtn: {
    width: 36, height: 36, borderRadius: 12, backgroundColor: "#ffffff",
    alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: C.border,
    shadowColor: "#000", shadowOpacity: 0.05, shadowRadius: 4, elevation: 1,
  },

  // Outbreak
  outbreakBanner: {
    borderRadius: 18, padding: 14, marginBottom: 12,
    borderWidth: 1,
  },
  outbreakRow: { flexDirection: "row", alignItems: "center", marginBottom: 10 },
  outbreakIcon: { width: 40, height: 40, borderRadius: 13, alignItems: "center", justifyContent: "center" },
  outbreakTitle: { fontSize: 13, fontWeight: "800", letterSpacing: -0.2 },
  outbreakMsg: { color: "#334155", fontSize: 12, marginTop: 2, fontWeight: "500" },
  outbreakItem: { marginTop: 10, paddingTop: 10, borderTopWidth: 1, borderTopColor: "rgba(0,0,0,0.06)" },
  outbreakSyndrome: { color: "#0f172a", fontWeight: "800", fontSize: 14, marginRight: 8 },

  // Hero card
  heroCard: {
    backgroundColor: "#ffffff", borderRadius: 24, padding: 20, marginTop: 4,
    borderWidth: 1, borderColor: C.border, borderLeftWidth: 5,
    shadowColor: "#000", shadowOpacity: 0.07, shadowRadius: 12, elevation: 3,
  },
  heroTopRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  statusIconWrap: { width: 42, height: 42, borderRadius: 14, alignItems: "center", justifyContent: "center" },
  heroSubHeader: { color: C.sub, fontSize: 12, fontWeight: "600" },
  statusTitle: { fontSize: 16, fontWeight: "800", marginTop: 2 },
  heroNumberSection: { marginVertical: 14 },
  mainNumber: { fontSize: 56, fontWeight: "900", color: C.text, letterSpacing: -2, lineHeight: 60 },
  mainNumberUnit: { color: C.sub, fontSize: 14, fontWeight: "600", marginTop: 2 },
  confidenceText: { color: C.textMuted, fontSize: 12, fontWeight: "600", marginTop: 6, fontStyle: "italic" },
  heroFooter: {
    borderTopWidth: 1, borderTopColor: C.divider, paddingTop: 12,
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
  },
  deltaBadge: { flexDirection: "row", alignItems: "center", paddingVertical: 5, paddingHorizontal: 10, borderRadius: 10 },
  deltaText: { fontSize: 12, fontWeight: "800" },
  statusReason: { color: C.sub, fontSize: 11, fontWeight: "500", maxWidth: 180, textAlign: "right" },

  statsRow: { flexDirection: "row", marginTop: 12 },

  alertBanner: {
    backgroundColor: C.redBg, borderRadius: 18, padding: 14, marginTop: 14,
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    borderWidth: 1, borderColor: "#fecaca",
  },
  alertBannerLeft: { flexDirection: "row", alignItems: "center", flex: 1 },
  alertIconCircle: {
    width: 40, height: 40, borderRadius: 13, backgroundColor: "#fee2e2",
    alignItems: "center", justifyContent: "center",
  },
  alertBannerTitle: { color: C.red, fontWeight: "800", fontSize: 14 },
  alertBannerSub: { color: C.textSecondary, fontSize: 12, marginTop: 2 },

  sectionMeta: { color: C.sub, fontSize: 12, fontWeight: "600" },
  linkText: { color: C.primary, fontSize: 12, fontWeight: "800" },

  synRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingVertical: 8 },
  synLeft: { flexDirection: "row", alignItems: "center" },
  rankPill: { width: 28, height: 28, borderRadius: 8, alignItems: "center", justifyContent: "center" },
  rankText: { fontWeight: "800", fontSize: 12 },
  synName: { color: C.text, fontWeight: "800", fontSize: 14 },
  synRiskLabel: { fontSize: 11, marginTop: 2, fontWeight: "700" },
  synRight: { alignItems: "flex-end" },
  synPercent: { fontWeight: "800", fontSize: 14 },

  demandGrid: { flexDirection: "row", flexWrap: "wrap", gap: 10, marginTop: 6 },
  demandCard: {
    width: "48%", backgroundColor: "#ffffff", padding: 14, borderRadius: 16,
    borderWidth: 1, borderColor: C.border,
    shadowColor: "#000", shadowOpacity: 0.05, shadowRadius: 6, elevation: 2,
  },
  demandItemName: { color: C.sub, fontSize: 12, fontWeight: "700", marginBottom: 4, flex: 1 },
  demandUnits: { color: C.text, fontSize: 22, fontWeight: "900", marginTop: 4 },
  demandUnit: { color: C.sub, fontSize: 11, fontWeight: "600" },
  demandRange: { color: C.textMuted, fontSize: 10, fontWeight: "600", marginTop: 4 },

  logRow: { flexDirection: "row", alignItems: "center", paddingVertical: 5, borderBottomWidth: 1, borderBottomColor: C.divider },
  logLabel: { flex: 1, color: C.textSecondary, fontSize: 13, fontWeight: "600" },
  logValue: { fontSize: 14, fontWeight: "800" },
  notesSnippet: { flexDirection: "row", alignItems: "flex-start", marginTop: 10, padding: 10, backgroundColor: C.bgSubtle, borderRadius: 10 },
  notesSnippetText: { color: C.sub, fontSize: 12, fontWeight: "500", flex: 1 },
  loggedByText: { color: C.textMuted, fontSize: 11, fontWeight: "600", marginTop: 8, textAlign: "right" },
  notesBox: { marginTop: 8 },
  notesLabel: { color: C.sub, fontSize: 12, fontWeight: "700", marginBottom: 6 },
  notesInput: {
    backgroundColor: "#ffffff", borderRadius: 14, padding: 12, color: C.text,
    fontSize: 14, borderWidth: 1, borderColor: C.inputBorder,
    textAlignVertical: "top", minHeight: 80,
  },

  // Day detail modal
  modalBackdrop: { flex: 1, backgroundColor: "rgba(15,23,42,0.5)", justifyContent: "center", alignItems: "center", padding: 24 },
  dayModal: {
    width: "100%", maxWidth: 380, backgroundColor: "#fff", borderRadius: 22, padding: 20,
    borderWidth: 1, borderColor: C.border,
    shadowColor: "#000", shadowOpacity: 0.2, shadowRadius: 16, elevation: 8,
  },
  dayModalTitle: { color: C.text, fontSize: 17, fontWeight: "900" },
});
