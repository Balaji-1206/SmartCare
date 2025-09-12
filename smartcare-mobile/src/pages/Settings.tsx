// src/pages/Settings.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  StyleSheet,
  ActivityIndicator,
  Pressable,
  Animated,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Ionicons } from "@expo/vector-icons";
import { Card, ErrorBanner } from "../ui";
import { C, API_DEFAULT, getApiBase, setApiBase, getNurseName, setNurseName, setAuthed } from "../constants";

/**
 * Polished Settings screen — drop-in replacement
 *
 * - Ripple-free : activeOpacity / android_ripple set to remove Android green flash
 * - Theme preview, API test, Save, Export logs, Force rollover, Clear local settings, Logout
 */

const K_THEME = "smartcare_theme";
const K_REMEMBER = "smartcare_remember_device";

export default function SettingsScreen({ onLogout }: { onLogout: () => void }) {
  const [api, setApi] = useState(API_DEFAULT);
  const [nurse, setNurse] = useState("");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [themePreview, setThemePreview] = useState<"light" | "dark">("dark");
  const [remember, setRemember] = useState(true);

  const btnScale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    (async () => {
      const base = await getApiBase(API_DEFAULT);
      setApi(base);
      const n = (await getNurseName()) || "";
      setNurse(n);
      try {
        const t = (await AsyncStorage.getItem(K_THEME)) as "light" | "dark" | null;
        if (t) setThemePreview(t);
      } catch {}
      try {
        const r = (await AsyncStorage.getItem(K_REMEMBER)) ?? "1";
        setRemember(r === "1");
      } catch {}
    })();
  }, []);

  const pulse = (node: Animated.Value) =>
    Animated.sequence([Animated.timing(node, { toValue: 0.96, duration: 100, useNativeDriver: true }), Animated.timing(node, { toValue: 1, duration: 150, useNativeDriver: true })]);

  async function test() {
    try {
      setTesting(true);
      setStatusMsg("Testing API...");
      const base = api.replace(/\/+$/, "");
      const res = await fetch(`${base}/`);
      if (!res.ok) throw new Error(`${res.status}`);
      const j = await res.json();
      setStatusMsg(`OK — ${j?.app ?? "API reachable"}`);
      Animated.sequence([pulse(btnScale)]).start();
    } catch (err: any) {
      setStatusMsg(`Error: ${err?.message ?? "unreachable"}`);
      Alert.alert("SmartCare", `Error: ${err?.message ?? "cannot reach API"}`);
    } finally {
      setTesting(false);
    }
  }

  async function save() {
    try {
      setSaving(true);
      await setApiBase(api);
      await setNurseName(nurse || "Nurse");
      await AsyncStorage.setItem(K_THEME, themePreview);
      await AsyncStorage.setItem(K_REMEMBER, remember ? "1" : "0");
      setStatusMsg("Saved ✅");
      Animated.sequence([pulse(btnScale)]).start();
      Alert.alert("SmartCare", "Settings saved");
    } catch (e: any) {
      Alert.alert("SmartCare", "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function logout() {
    await setAuthed(false);
    onLogout();
  }

  async function clearLocalSettings() {
    Alert.alert("Confirm", "Clear saved API & nurse name? This does not touch backend.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Clear",
        style: "destructive",
        onPress: async () => {
          try {
            await AsyncStorage.removeItem("smartcare_api_base");
            await AsyncStorage.removeItem("smartcare_nurse_name");
            await AsyncStorage.removeItem(K_THEME);
            await AsyncStorage.removeItem(K_REMEMBER);
            setApi(API_DEFAULT);
            setNurse("");
            setThemePreview("dark");
            setRemember(true);
            setStatusMsg("Local settings cleared");
            Alert.alert("SmartCare", "Local settings cleared");
          } catch {
            Alert.alert("SmartCare", "Failed to clear storage");
          }
        },
      },
    ]);
  }

  async function doRollover() {
    try {
      setStatusMsg("Triggering rollover...");
      const base = api.replace(/\/+$/, "");
      const r = await fetch(`${base}/rollover`);
      if (!r.ok) throw new Error(`${r.status}`);
      const j = await r.json();
      setStatusMsg("Rollover finished");
      Alert.alert("SmartCare", `Rollover: ${JSON.stringify(j)}`);
    } catch (e: any) {
      setStatusMsg(`Rollover failed: ${e?.message ?? "error"}`);
      Alert.alert("SmartCare", `Rollover failed: ${e?.message ?? "error"}`);
    }
  }

  async function exportNurseLogs() {
    try {
      setStatusMsg("Fetching nurse logs...");
      const base = api.replace(/\/+$/, "");
      const r = await fetch(`${base}/debug/nurse-log`);
      if (!r.ok) throw new Error(`${r.status}`);
      const j = await r.json();
      const entries = Object.keys(j || {}).length;
      setStatusMsg(`Got ${entries} dated entries`);
      Alert.alert("Nurse logs", `Found ${entries} dated entries.\nOpen console for full dump.`);
      console.log("nurse-log:", j);
    } catch (e: any) {
      setStatusMsg(`Fetch failed: ${e?.message ?? "error"}`);
      Alert.alert("SmartCare", `Failed to fetch logs: ${e?.message ?? "error"}`);
    }
  }

  const themePreviewStyle =
    themePreview === "dark"
      ? { backgroundColor: "#0b1220", color: "#fff", borderColor: C.border }
      : { backgroundColor: "#ffffff", color: "#0b1220", borderColor: "#e5e7eb" };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 16 }}>
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Ionicons name="settings-outline" size={28} color={C.primary} />
              <View style={{ marginLeft: 12 }}>
                <Text style={styles.headerTitle}>Settings</Text>
                <Text style={styles.headerSub}>Configure API, nurse & utilities</Text>
              </View>
            </View>
            <Animated.View style={{ transform: [{ scale: btnScale }] }}>
              <TouchableOpacity
                onPress={save}
                activeOpacity={0.9}
                style={{
                  backgroundColor: C.primary,
                  paddingHorizontal: 14,
                  paddingVertical: 10,
                  borderRadius: 10,
                }}
              >
                {saving ? <ActivityIndicator color="#fff" /> : <Text style={{ color: "#fff", fontWeight: "800" }}>Save</Text>}
              </TouchableOpacity>
            </Animated.View>
          </View>

          <ErrorBanner msg={statusMsg} />

          {/* API Card */}
          <Card title="API Base URL">
            <Text style={{ color: C.sub, marginBottom: 8 }}>Where the backend lives (device reachable)</Text>

            <View style={styles.inputRow}>
              <Ionicons name="link-outline" size={18} color={C.sub} style={{ marginRight: 8 }} />
              <TextInput
                value={api}
                onChangeText={(v) => setApi(v)}
                autoCapitalize="none"
                placeholder={API_DEFAULT}
                placeholderTextColor={C.sub}
                style={styles.input}
                accessibilityLabel="API base"
              />
            </View>

            <View style={styles.rowActions}>
              <TouchableOpacity onPress={test} activeOpacity={0.9} style={[styles.actionBtn, { backgroundColor: testing ? "#334155" : C.primary }]}>
                {testing ? <ActivityIndicator color="#fff" /> : <Text style={styles.actionBtnText}>Test API</Text>}
              </TouchableOpacity>

              <TouchableOpacity onPress={() => { setApi(API_DEFAULT); setStatusMsg(null); }} activeOpacity={0.9} style={[styles.actionBtn, styles.ghost]}>
                <Text style={[styles.actionBtnText, { color: C.text }]}>Reset</Text>
              </TouchableOpacity>
            </View>

            {!!statusMsg && <Text style={{ color: C.sub, marginTop: 10 }}>{statusMsg}</Text>}
          </Card>

          {/* Nurse Card */}
          <Card title="Nurse">
            <Text style={{ color: C.sub, marginBottom: 6 }}>Nurse display name</Text>
            <View style={styles.inputRow}>
              <Ionicons name="person-outline" size={18} color={C.sub} style={{ marginRight: 8 }} />
              <TextInput value={nurse} onChangeText={setNurse} placeholder="Meena" placeholderTextColor={C.sub} style={styles.input} />
            </View>

            <View style={{ flexDirection: "row", alignItems: "center", marginTop: 12 }}>
              <Pressable onPress={() => setRemember((s) => !s)} android_ripple={{ color: "transparent" }} style={{ padding: 8 }}>
                <Ionicons name={remember ? "checkmark-circle" : "ellipse-outline"} size={20} color={remember ? C.primary : C.sub} />
              </Pressable>
              <Text style={{ color: C.sub, marginLeft: 8 }}>Remember nurse name on this device</Text>
            </View>

            <View style={{ marginTop: 12 }}>
              <View style={{ flexDirection: "row", alignItems: "center" }}>
                <Text style={{ color: C.sub, marginRight: 8 }}>Theme</Text>
                <TouchableOpacity activeOpacity={0.9} onPress={() => setThemePreview("light")} style={[styles.themePill, themePreview === "light" && styles.themePillActive]}>
                  <Text style={{ fontWeight: "700", color: themePreview === "light" ? "#fff" : C.text }}>Light</Text>
                </TouchableOpacity>
                <TouchableOpacity activeOpacity={0.9} onPress={() => setThemePreview("dark")} style={[styles.themePill, themePreview === "dark" && styles.themePillActive, { marginLeft: 8 }]}>
                  <Text style={{ fontWeight: "700", color: themePreview === "dark" ? "#fff" : C.text }}>Dark</Text>
                </TouchableOpacity>

                <View style={{ marginLeft: 12 }}>
                  <View style={{ borderRadius: 8, paddingVertical: 6, paddingHorizontal: 10, borderWidth: 1, borderColor: themePreviewStyle.borderColor, backgroundColor: themePreviewStyle.backgroundColor }}>
                    <Text style={{ color: themePreviewStyle.color, fontWeight: "700", textTransform: "capitalize" }}>{themePreview} preview</Text>
                  </View>
                </View>
              </View>
            </View>
          </Card>

          {/* Utilities */}
          <Card title="Utilities">
            <TouchableOpacity activeOpacity={0.9} onPress={exportNurseLogs} style={styles.utilityRow}>
              <View style={styles.utilityLeft}>
                <Ionicons name="download-outline" size={18} color={C.primary} style={{ marginRight: 10 }} />
                <View>
                  <Text style={{ color: C.text, fontWeight: "700" }}>Export nurse logs (debug)</Text>
                  <Text style={{ color: C.sub, fontSize: 12 }}>Download logs for offline review</Text>
                </View>
              </View>
              <Text style={{ color: C.sub }}>Quick</Text>
            </TouchableOpacity>

            <TouchableOpacity activeOpacity={0.9} onPress={doRollover} style={[styles.utilityRow, { marginTop: 8 }]}>
              <View style={styles.utilityLeft}>
                <Ionicons name="repeat-outline" size={18} color={C.yellow} style={{ marginRight: 10 }} />
                <View>
                  <Text style={{ color: C.text, fontWeight: "700" }}>Force rollover (append yesterday)</Text>
                  <Text style={{ color: C.sub, fontSize: 12 }}>Run server-side rollover</Text>
                </View>
              </View>
              <Text style={{ color: C.sub }}>Server</Text>
            </TouchableOpacity>

            <TouchableOpacity activeOpacity={0.9} onPress={clearLocalSettings} style={[styles.utilityRow, { marginTop: 8, backgroundColor: "#fff7f7", borderWidth: 1, borderColor: "#fecaca" }]}>
              <View style={styles.utilityLeft}>
                <Ionicons name="trash-outline" size={18} color={"#b91c1c"} style={{ marginRight: 10 }} />
                <View>
                  <Text style={{ color: "#b91c1c", fontWeight: "700" }}>Clear saved settings</Text>
                  <Text style={{ color: C.sub, fontSize: 12 }}>Remove API & nurse name (local only)</Text>
                </View>
              </View>
              <Text style={{ color: C.sub }}>Local</Text>
            </TouchableOpacity>
          </Card>

          <View style={{ marginTop: 12 }}>
            <TouchableOpacity activeOpacity={0.9} onPress={logout} style={{ backgroundColor: "#111827", paddingVertical: 14, borderRadius: 12, alignItems: "center", borderWidth: 1, borderColor: C.border }}>
              <Text style={{ color: "#fff", fontWeight: "800", fontSize: 15 }}>Logout</Text>
            </TouchableOpacity>
          </View>

          <View style={{ alignItems: "center", marginTop: 18, marginBottom: 36 }}>
            <Text style={{ color: C.sub }}>SmartCare • v0.3.0</Text>
            <Text style={{ color: C.sub, fontSize: 12, marginTop: 6 }}>Backend: {api.replace(/https?:\/\//, "")}</Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  headerLeft: { flexDirection: "row", alignItems: "center" },
  headerTitle: { color: C.text, fontSize: 22, fontWeight: "800" },
  headerSub: { color: C.sub, marginTop: 2 },

  inputRow: { flexDirection: "row", alignItems: "center", backgroundColor: "#071025", paddingHorizontal: 12, paddingVertical: Platform.OS === "ios" ? 12 : 8, borderRadius: 12, borderWidth: 1, borderColor: C.border },
  input: { color: "#fff", flex: 1, fontSize: 15 },

  rowActions: { flexDirection: "row", gap: 8, marginTop: 12, alignItems: "center" },
  actionBtn: { flex: 1, paddingVertical: 12, borderRadius: 12, alignItems: "center", justifyContent: "center" },
  actionBtnText: { color: "#fff", fontWeight: "800" },
  ghost: { backgroundColor: C.chip },

  themePill: { paddingVertical: 8, paddingHorizontal: 12, borderRadius: 999, backgroundColor: C.chip },
  themePillActive: { backgroundColor: C.primary },

  utilityRow: { backgroundColor: C.card, padding: 12, borderRadius: 12, flexDirection: "row", justifyContent: "space-between", alignItems: "center", borderWidth: 1, borderColor: C.border },
  utilityLeft: { flexDirection: "row", alignItems: "center" },

  // small consistent card spacing
  controlRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },

  // visuals
  centered: { alignItems: "center" },

  // snack / status (use ErrorBanner above for top message)
});
