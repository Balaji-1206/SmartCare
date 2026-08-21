// src/pages/Settings.tsx
import React, { useCallback, useEffect, useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  StyleSheet,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Card, SectionTitle, Button, ErrorBanner, InfoRow, Divider, Pill } from "../ui";
import {
  C,
  API_DEFAULT,
  getApiBase,
  setApiBase,
  getNurseName,
  setNurseName,
  setAuthed,
  getLatLon,
  setLatLon,
} from "../constants";
import { apiGet, apiPost } from "../api";
import { flush } from "../offlineQueue";
import AsyncStorage from "@react-native-async-storage/async-storage";

const APP_VERSION = "1.1.0";
const APP_BUILD   = "2026.08.22";

export default function SettingsScreen({ onLogout }: { onLogout: () => void }) {
  const [api, setApi] = useState(API_DEFAULT);
  const [nurse, setNurse] = useState("");
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [pingResult, setPingResult] = useState<{ msg: string; ok: boolean } | null>(null);
  const [queueCount, setQueueCount] = useState<number>(0);
  const [apiFocus, setApiFocus] = useState(false);
  const [nurseFocus, setNurseFocus] = useState(false);

  // Weather & Coordinates
  const [lat, setLat] = useState("13.0827");
  const [lon, setLon] = useState("80.2707");
  const [temp, setTemp] = useState("");
  const [rainfall, setRainfall] = useState("");
  const [humidity, setHumidity] = useState("");
  const [fetchingWeather, setFetchingWeather] = useState(false);
  const [savingWeather, setSavingWeather] = useState(false);

  useEffect(() => {
    (async () => {
      const base = await getApiBase(API_DEFAULT);
      setApi(base);
      const n = await getNurseName();
      setNurse(n || "");
      const coords = await getLatLon();
      if (coords.lat) setLat(coords.lat);
      if (coords.lon) setLon(coords.lon);
      checkQueueCount();
      loadCurrentWeather(base);
    })();
  }, []);

  const loadCurrentWeather = async (base: string) => {
    try {
      const w = await apiGet<{ temperature?: number; rainfall?: number; humidity?: number }>(base, "/weather/today");
      if (w) {
        if (w.temperature != null) setTemp(String(w.temperature));
        if (w.rainfall != null) setRainfall(String(w.rainfall));
        if (w.humidity != null) setHumidity(String(w.humidity));
      }
    } catch {}
  };

  const checkQueueCount = async () => {
    try {
      const raw = await AsyncStorage.getItem("smartcare_offline_queue");
      const q = raw ? JSON.parse(raw) : [];
      setQueueCount(Array.isArray(q) ? q.length : 0);
    } catch {
      setQueueCount(0);
    }
  };

  function showMsg(text: string, type: "success" | "error" | "info" = "success") {
    setStatusMsg({ text, type });
    setTimeout(() => setStatusMsg(null), 3500);
  }

  async function testConnection() {
    try {
      setTesting(true);
      setPingResult(null);
      const base = api.replace(/\/+$/, "");
      const t0 = Date.now();
      const res = await fetch(`${base}/`);
      const latency = Date.now() - t0;
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = await res.json();
      setPingResult({ msg: `✓ Connected to ${j?.app || "API"} — ${latency}ms`, ok: true });
    } catch (err: any) {
      setPingResult({ msg: `✗ Cannot reach ${api}`, ok: false });
    } finally {
      setTesting(false);
    }
  }

  async function saveConfig() {
    try {
      setSaving(true);
      await setApiBase(api.trim());
      await setNurseName(nurse.trim() || "On-duty Nurse");
      await setLatLon(lat.trim(), lon.trim());
      showMsg("Settings saved successfully");
    } catch (e: any) {
      showMsg("Failed to save settings", "error");
    } finally {
      setSaving(false);
    }
  }

  async function handleFetchWeather() {
    if (!lat || !lon) {
      showMsg("Please enter valid Latitude & Longitude", "error");
      return;
    }
    try {
      setFetchingWeather(true);
      await setLatLon(lat.trim(), lon.trim());
      const res = await apiPost<any>(api, "/weather/fetch", {
        lat: parseFloat(lat),
        lon: parseFloat(lon),
        units: "metric",
      });
      if (res.ok && res.applied) {
        if (res.applied.temperature != null) setTemp(String(res.applied.temperature));
        if (res.applied.rainfall != null) setRainfall(String(res.applied.rainfall));
        if (res.applied.humidity != null) setHumidity(String(res.applied.humidity));
        showMsg("Live weather telemetry pulled & applied!");
      }
    } catch (e: any) {
      showMsg(e?.message ?? "Failed to fetch live weather (check API key or use manual override)", "error");
    } finally {
      setFetchingWeather(false);
    }
  }

  async function handleSaveManualWeather() {
    try {
      setSavingWeather(true);
      const today = new Date().toISOString().slice(0, 10);
      const payload = {
        date: today,
        temperature: temp ? parseFloat(temp) : null,
        rainfall: rainfall ? parseFloat(rainfall) : null,
        humidity: humidity ? parseFloat(humidity) : null,
      };
      const res = await apiPost<any>(api, "/weather/upsert", payload);
      if (res.ok) {
        showMsg("Weather conditions saved and model re-calibrated!");
      }
    } catch (e: any) {
      showMsg(e?.message ?? "Failed to save weather conditions", "error");
    } finally {
      setSavingWeather(false);
    }
  }

  async function syncOfflineQueue() {
    try {
      setSyncing(true);
      const res = await flush(api);
      await checkQueueCount();
      if (!res.ok) {
        showMsg("Device offline — queue will sync when connected", "error");
      } else if (res.count === 0) {
        showMsg("Queue is empty. All data is synced!", "success");
      } else {
        showMsg(`Synced ${res.count} record(s). ${res.remaining} remaining.`);
      }
    } catch (e: any) {
      showMsg(e?.message ?? "Error syncing", "error");
    } finally {
      setSyncing(false);
    }
  }

  async function clearAllLocalData() {
    Alert.alert(
      "Reset Local Data",
      "This will clear all locally cached queue items and your login session. Server data is not affected.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset",
          style: "destructive",
          onPress: async () => {
            try {
              await AsyncStorage.multiRemove([
                "smartcare_offline_queue",
                "smartcare_api_base",
                "smartcare_nurse_name",
                "smartcare_weather_lat",
                "smartcare_weather_lon",
              ]);
              setQueueCount(0);
              showMsg("Local data cleared. App will use defaults.");
              setApi(API_DEFAULT);
              setNurse("");
            } catch {
              showMsg("Failed to clear data", "error");
            }
          },
        },
      ]
    );
  }

  async function handleLogout() {
    Alert.alert("Sign Out", "Exit the station? Your server data is saved.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Sign Out",
        style: "destructive",
        onPress: async () => {
          await setAuthed(false);
          onLogout();
        },
      },
    ]);
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Profile Header */}
        <View style={styles.profileHeader}>
          <View style={styles.avatarCircle}>
            <Text style={styles.avatarInitials}>
              {nurse ? nurse.trim().slice(0, 2).toUpperCase() : "PH"}
            </Text>
          </View>
          <View style={{ marginLeft: 14, flex: 1 }}>
            <Text style={styles.nurseName}>{nurse || "Duty Nurse"}</Text>
            <View style={{ flexDirection: "row", alignItems: "center", marginTop: 4 }}>
              <Pill text="Active Station" bg={C.greenBg} textColor={C.green} icon="wifi" size="sm" />
            </View>
          </View>
        </View>

        {statusMsg && <ErrorBanner msg={statusMsg.text} type={statusMsg.type} />}

        {/* Server Config */}
        <SectionTitle title="Server Configuration" icon="server-outline" />
        <Card style={{ marginTop: 6 }}>
          <Text style={styles.labelSub}>FastAPI backend base URL</Text>
          <View style={[styles.inputRow, apiFocus && styles.inputFocus]}>
            <Ionicons name="link-outline" size={17} color={apiFocus ? C.primary : C.sub} style={{ marginRight: 8 }} />
            <TextInput
              value={api}
              onChangeText={setApi}
              autoCapitalize="none"
              placeholder={API_DEFAULT}
              placeholderTextColor={C.textMuted}
              style={styles.input}
              onFocus={() => setApiFocus(true)}
              onBlur={() => setApiFocus(false)}
            />
          </View>

          <View style={{ flexDirection: "row", gap: 10, marginTop: 12 }}>
            <View style={{ flex: 1 }}>
              <Button title={testing ? "Pinging..." : "Test Connection"} variant="outline" loading={testing} onPress={testConnection} icon="pulse-outline" size="sm" />
            </View>
            <View style={{ flex: 1 }}>
              <Button title="Reset to Default" variant="secondary" size="sm"
                onPress={() => { setApi(API_DEFAULT); setPingResult(null); }} />
            </View>
          </View>

          {pingResult && (
            <View style={[styles.pingResult, { backgroundColor: pingResult.ok ? C.greenBg : C.redBg, borderColor: pingResult.ok ? "#bbf7d0" : "#fecaca" }]}>
              <Text style={{ color: pingResult.ok ? C.green : C.red, fontWeight: "700", fontSize: 13 }}>{pingResult.msg}</Text>
            </View>
          )}
        </Card>

        {/* Nurse Profile */}
        <SectionTitle title="Nurse Profile" icon="person-circle-outline" />
        <Card style={{ marginTop: 6 }}>
          <Text style={styles.labelSub}>Your name as logged in triage reports</Text>
          <View style={[styles.inputRow, nurseFocus && styles.inputFocus]}>
            <Ionicons name="person-outline" size={17} color={nurseFocus ? C.primary : C.sub} style={{ marginRight: 8 }} />
            <TextInput
              value={nurse}
              onChangeText={setNurse}
              placeholder="e.g. Sister Meena"
              placeholderTextColor={C.textMuted}
              style={styles.input}
              onFocus={() => setNurseFocus(true)}
              onBlur={() => setNurseFocus(false)}
            />
          </View>
          <Button title={saving ? "Saving..." : "Save Settings"} loading={saving} onPress={saveConfig} icon="save-outline" style={{ marginTop: 12 }} />
        </Card>

        {/* Location & Weather Telemetry */}
        <SectionTitle title="Weather & Environmental Telemetry" icon="partly-sunny-outline" />
        <Card style={{ marginTop: 6 }}>
          <Text style={styles.labelSub}>Station Coordinates (used for ML surge correlation)</Text>
          <View style={{ flexDirection: "row", gap: 10, marginBottom: 12 }}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldHeader}>Latitude</Text>
              <View style={styles.inputRow}>
                <TextInput
                  value={lat}
                  onChangeText={setLat}
                  placeholder="13.0827"
                  placeholderTextColor={C.textMuted}
                  keyboardType="numeric"
                  style={styles.input}
                />
              </View>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldHeader}>Longitude</Text>
              <View style={styles.inputRow}>
                <TextInput
                  value={lon}
                  onChangeText={setLon}
                  placeholder="80.2707"
                  placeholderTextColor={C.textMuted}
                  keyboardType="numeric"
                  style={styles.input}
                />
              </View>
            </View>
          </View>

          <Button
            title={fetchingWeather ? "Pulling Live Weather..." : "Fetch Live OpenWeather"}
            variant="outline"
            size="sm"
            loading={fetchingWeather}
            onPress={handleFetchWeather}
            icon="cloud-download-outline"
            style={{ marginBottom: 14 }}
          />

          <Divider />

          <Text style={[styles.fieldHeader, { marginTop: 12 }]}>Today's Climate Conditions</Text>
          <View style={{ flexDirection: "row", gap: 8, marginTop: 6, marginBottom: 12 }}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldSubLabel}>Temp (°C)</Text>
              <View style={styles.inputRow}>
                <TextInput
                  value={temp}
                  onChangeText={setTemp}
                  placeholder="31"
                  placeholderTextColor={C.textMuted}
                  keyboardType="numeric"
                  style={styles.input}
                />
              </View>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldSubLabel}>Rain (mm)</Text>
              <View style={styles.inputRow}>
                <TextInput
                  value={rainfall}
                  onChangeText={setRainfall}
                  placeholder="0"
                  placeholderTextColor={C.textMuted}
                  keyboardType="numeric"
                  style={styles.input}
                />
              </View>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldSubLabel}>Humidity (%)</Text>
              <View style={styles.inputRow}>
                <TextInput
                  value={humidity}
                  onChangeText={setHumidity}
                  placeholder="70"
                  placeholderTextColor={C.textMuted}
                  keyboardType="numeric"
                  style={styles.input}
                />
              </View>
            </View>
          </View>

          <Button
            title={savingWeather ? "Saving Telemetry..." : "Update Weather Overrides"}
            loading={savingWeather}
            onPress={handleSaveManualWeather}
            icon="checkmark-done-outline"
            size="sm"
          />
        </Card>

        {/* Offline Sync */}
        <SectionTitle title="Offline Data Sync" icon="cloud-upload-outline" />
        <Card style={{ marginTop: 6 }}>
          <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <Text style={styles.cardInlineTitle}>Offline Queue</Text>
            <View style={[styles.badge, queueCount > 0 ? { backgroundColor: C.yellowBg } : { backgroundColor: C.greenBg }]}>
              <Text style={{ color: queueCount > 0 ? C.yellow : C.green, fontWeight: "800", fontSize: 13 }}>
                {queueCount} {queueCount === 1 ? "item" : "items"} pending
              </Text>
            </View>
          </View>
          <Text style={styles.labelSub}>
            When offline, nurse logs and inventory edits are stored locally and synced automatically. You can also force-sync manually.
          </Text>
          <View style={{ flexDirection: "row", gap: 10, marginTop: 12 }}>
            <View style={{ flex: 2 }}>
              <Button
                title={syncing ? "Syncing..." : "Force Sync Now"}
                onPress={syncOfflineQueue}
                loading={syncing}
                icon="sync-outline"
                variant={queueCount > 0 ? "primary" : "secondary"}
              />
            </View>
            <View style={{ flex: 1 }}>
              <Button title="Refresh" variant="secondary" onPress={checkQueueCount} icon="refresh-outline" />
            </View>
          </View>
        </Card>

        {/* App Info */}
        <SectionTitle title="About SmartCare" icon="information-circle-outline" />
        <Card style={{ marginTop: 6 }}>
          <InfoRow icon="code-slash-outline" label="Version" value={APP_VERSION} />
          <Divider />
          <InfoRow icon="calendar-outline" label="Build Date" value={APP_BUILD} />
          <Divider />
          <InfoRow icon="phone-portrait-outline" label="Platform" value={Platform.OS === "web" ? "Web Browser" : Platform.OS === "ios" ? "iOS" : "Android"} />
          <Divider />
          <InfoRow icon="hardware-chip-outline" label="ML Models" value="GBM v1.0 + Outbreak Monitor" />
          <Divider />
          <InfoRow icon="server-outline" label="Backend" value="FastAPI + Python 3.13" />
        </Card>

        {/* Danger Zone */}
        <SectionTitle title="Danger Zone" icon="warning-outline" />
        <Card style={{ marginTop: 6 }}>
          <TouchableOpacity onPress={clearAllLocalData} activeOpacity={0.8} style={styles.dangerRow}>
            <View style={{ flexDirection: "row", alignItems: "center", flex: 1 }}>
              <View style={[styles.dangerIcon, { backgroundColor: C.yellowBg }]}>
                <Ionicons name="trash-outline" size={18} color={C.yellow} />
              </View>
              <View style={{ marginLeft: 12 }}>
                <Text style={styles.dangerTitle}>Clear Local Cache</Text>
                <Text style={styles.dangerSub}>Wipes offline queue and stored credentials</Text>
              </View>
            </View>
            <Ionicons name="chevron-forward" size={16} color={C.sub} />
          </TouchableOpacity>
        </Card>

        {/* Sign Out */}
        <View style={{ marginTop: 20 }}>
          <Button
            title="Sign Out of Station"
            variant="danger"
            onPress={handleLogout}
            icon="log-out-outline"
          />
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>SmartCare PHC Intelligence Suite</Text>
          <Text style={styles.footerSub}>v{APP_VERSION} • Powered by Machine Learning</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  scroll: { padding: 16, paddingBottom: 50 },
  profileHeader: {
    flexDirection: "row", alignItems: "center",
    backgroundColor: "#ffffff", borderRadius: 20, padding: 16, marginBottom: 8,
    borderWidth: 1, borderColor: C.border,
    shadowColor: "#000", shadowOpacity: 0.05, shadowRadius: 8, elevation: 2,
  },
  avatarCircle: {
    width: 52, height: 52, borderRadius: 18, backgroundColor: C.primaryBg,
    alignItems: "center", justifyContent: "center", borderWidth: 2, borderColor: "#bae6fd",
  },
  avatarInitials: { color: C.primary, fontWeight: "900", fontSize: 18 },
  nurseName: { color: C.text, fontSize: 18, fontWeight: "800" },
  labelSub: { color: C.sub, fontSize: 12, marginBottom: 8 },
  fieldHeader: { color: C.textSecondary, fontSize: 12, fontWeight: "700", marginBottom: 4 },
  fieldSubLabel: { color: C.sub, fontSize: 11, fontWeight: "600", marginBottom: 4 },
  inputRow: {
    flexDirection: "row", alignItems: "center", backgroundColor: "#ffffff",
    borderRadius: 14, paddingHorizontal: 14, paddingVertical: Platform.OS === "ios" ? 14 : 10,
    borderWidth: 1, borderColor: C.inputBorder,
  },
  inputFocus: { borderColor: C.inputFocusBorder },
  input: { flex: 1, color: C.text, fontSize: 14, fontWeight: "600" },
  pingResult: {
    padding: 10, borderRadius: 12, marginTop: 10, borderWidth: 1,
  },
  cardInlineTitle: { color: C.text, fontSize: 15, fontWeight: "800" },
  badge: { paddingHorizontal: 12, paddingVertical: 5, borderRadius: 10 },
  dangerRow: {
    flexDirection: "row", justifyContent: "space-between",
    alignItems: "center", paddingVertical: 4,
  },
  dangerIcon: { width: 38, height: 38, borderRadius: 12, alignItems: "center", justifyContent: "center" },
  dangerTitle: { color: C.text, fontSize: 14, fontWeight: "800" },
  dangerSub: { color: C.sub, fontSize: 12, marginTop: 2 },
  footer: { alignItems: "center", marginTop: 28 },
  footerText: { color: C.textMuted, fontSize: 12, fontWeight: "700" },
  footerSub: { color: C.textMuted, fontSize: 11, marginTop: 2 },
});
