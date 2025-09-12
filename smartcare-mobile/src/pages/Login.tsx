// src/pages/Login.tsx
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
  KeyboardAvoidingView,
  Platform,
  Animated,
  ActivityIndicator,
  Pressable,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Card, ErrorBanner } from "../ui";
import {
  C,
  API_DEFAULT,
  getApiBase,
  setApiBase,
  getNurseName,
  setNurseName,
  setAuthed,
} from "../constants";

const K_THEME = "smartcare_theme";
const K_REMEMBER = "smartcare_remember_device";

export default function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [api, setApi] = useState(API_DEFAULT);
  const [name, setName] = useState("");
  const [loadingTest, setLoadingTest] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testStatus, setTestStatus] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [remember, setRemember] = useState<boolean>(true);

  const logoScale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(logoScale, { toValue: 1.04, duration: 1000, useNativeDriver: true }),
        Animated.timing(logoScale, { toValue: 1.0, duration: 1000, useNativeDriver: true }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [logoScale]);

  const [snack, setSnack] = useState<{ type: "ok" | "err"; text: string } | null>(null);
  useEffect(() => {
    if (!snack) return;
    const t = setTimeout(() => setSnack(null), 2500);
    return () => clearTimeout(t);
  }, [snack]);

  useEffect(() => {
    (async () => {
      try {
        const base = await getApiBase(API_DEFAULT);
        setApi(base);
        const saved = (await getNurseName()) || "";
        setName(saved);
        const themeSaved = (await AsyncStorage.getItem(K_THEME)) as "light" | "dark" | null;
        if (themeSaved) setTheme(themeSaved);
        const rem = (await AsyncStorage.getItem(K_REMEMBER)) ?? "1";
        setRemember(rem === "1");
      } catch {}
    })();
  }, []);

  async function testApi() {
    setTestStatus(null);
    setErr(null);
    setLoadingTest(true);
    try {
      const url = api.replace(/\/+$/, "") || API_DEFAULT;
      const r = await fetch(`${url}/`);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const j = await r.json();
      const msg = j?.app ? `OK — ${j.app}` : "API reachable";
      setTestStatus(msg);
      setSnack({ type: "ok", text: "API reachable" });
    } catch (e: any) {
      const m = e?.message ?? "Cannot reach API";
      setErr(m);
      setTestStatus(null);
      setSnack({ type: "err", text: "API unreachable" });
      Alert.alert("SmartCare", m);
    } finally {
      setLoadingTest(false);
    }
  }

  async function onLogin() {
    setErr(null);
    if (!name.trim()) {
      setErr("Please enter your name");
      setSnack({ type: "err", text: "Enter your name" });
      return;
    }
    try {
      setSaving(true);
      await setApiBase(api);
      if (remember) await setNurseName(name.trim());
      else await setNurseName("");
      await setAuthed(true);
      await AsyncStorage.setItem(K_REMEMBER, remember ? "1" : "0");
      await AsyncStorage.setItem(K_THEME, theme);
      setSnack({ type: "ok", text: "Saved ✓" });
      onLoggedIn();
    } catch (e: any) {
      setSaving(false);
      const m = e?.message ?? "Failed to save";
      setErr(m);
      setSnack({ type: "err", text: "Save failed" });
      Alert.alert("SmartCare", m);
    }
  }

  function resetToDefault() {
    setApi(API_DEFAULT);
    setTestStatus(null);
    setErr(null);
    setSnack({ type: "ok", text: "Reset" });
  }

  const [apiFocus, setApiFocus] = useState(false);
  const [nameFocus, setNameFocus] = useState(false);

  return (
    <SafeAreaView style={[styles.safe, theme === "light" ? styles.lightBG : null]}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          <View style={styles.center}>
            <Animated.View style={[styles.logoWrap, { transform: [{ scale: logoScale }] }]}>
              <View style={styles.logoInner}>
                <Ionicons name="heart-circle-outline" size={56} color={C.primary} accessibilityLabel="SmartCare logo" />
              </View>
            </Animated.View>

            <Text style={styles.title}>SmartCare</Text>
            <Text style={styles.subtitle}>Lightweight clinic dashboard — sign in to continue</Text>
          </View>

          <ErrorBanner msg={err} />

          <Card style={{ padding: 16 }}>
            <View style={styles.rowBetween}>
              <Text style={styles.label}>API Base URL</Text>
              <Pressable
                onPress={() => {
                  const t = theme === "dark" ? "light" : "dark";
                  setTheme(t);
                  AsyncStorage.setItem(K_THEME, t).catch(() => {});
                  setSnack({ type: "ok", text: `Theme: ${t}` });
                }}
                android_ripple={{ color: "transparent" }}
              >
                <Ionicons name={theme === "dark" ? "moon" : "sunny"} size={18} color={C.sub} />
              </Pressable>
            </View>

            <View style={[styles.inputRow, apiFocus ? styles.inputFocus : null]}>
              <Ionicons name="server-outline" size={18} color={C.sub} style={{ marginRight: 8 }} />
              <TextInput
                value={api}
                onChangeText={setApi}
                autoCapitalize="none"
                placeholder={API_DEFAULT}
                placeholderTextColor={C.sub}
                keyboardType="url"
                style={[styles.input, { flex: 1 }]}
                onFocus={() => setApiFocus(true)}
                onBlur={() => setApiFocus(false)}
                accessibilityLabel="API base url"
              />
            </View>

            <View style={styles.controlRow}>
              <TouchableOpacity
                onPress={testApi}
                disabled={loadingTest}
                style={[styles.btn, { flex: 1, backgroundColor: loadingTest ? "#334155" : C.primary }]}
                activeOpacity={0.9}
              >
                {loadingTest ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnText}>Test API</Text>}
              </TouchableOpacity>

              <TouchableOpacity
                onPress={resetToDefault}
                style={[styles.btn, styles.ghostBtn]}
                activeOpacity={0.9}
              >
                <Text style={[styles.btnText, { color: C.text }]}>Reset</Text>
              </TouchableOpacity>
            </View>

            {testStatus && <Text style={{ color: C.green, marginTop: 10, fontWeight: "700" }}>{testStatus}</Text>}
          </Card>

          <Card style={{ padding: 16, marginTop: 12 }}>
            <View style={styles.rowBetween}>
              <Text style={styles.label}>Your Details</Text>
              <Pressable
                onPress={() => {
                  setRemember((r) => !r);
                  AsyncStorage.setItem(K_REMEMBER, !remember ? "1" : "0").catch(() => {});
                }}
                android_ripple={{ color: "transparent" }}
              >
                <Ionicons
                  name={remember ? "checkmark-circle" : "ellipse-outline"}
                  size={18}
                  color={remember ? C.primary : C.sub}
                />
              </Pressable>
            </View>

            <View style={[styles.inputRow, nameFocus ? styles.inputFocus : null]}>
              <Ionicons name="person-circle-outline" size={18} color={C.sub} style={{ marginRight: 8 }} />
              <TextInput
                value={name}
                onChangeText={setName}
                placeholder="Nurse name (e.g. Meena)"
                placeholderTextColor={C.sub}
                style={[styles.input, { flex: 1 }]}
                onFocus={() => setNameFocus(true)}
                onBlur={() => setNameFocus(false)}
                accessibilityLabel="Nurse display name"
              />
            </View>
          </Card>

          <View style={{ marginTop: 18 }}>
            <TouchableOpacity
              onPress={onLogin}
              disabled={saving}
              style={[styles.primaryBtn, { backgroundColor: saving ? "#1e293b" : C.primary }]}
              activeOpacity={0.9}
            >
              <Text style={{ color: "#fff", fontWeight: "900", fontSize: 16 }}>{saving ? "Saving…" : "Continue"}</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.footer}>
            <Text style={{ color: C.sub }}>Need help? Check your API URL & server</Text>
            <Text style={{ color: C.sub, marginTop: 8 }}>SmartCare • v0.3.0</Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {snack && (
        <View
          style={[
            styles.snack,
            snack.type === "ok"
              ? { backgroundColor: "#ecfdf5", borderColor: C.green }
              : { backgroundColor: "#fff7f7", borderColor: C.red },
          ]}
        >
          <Text style={{ color: snack.type === "ok" ? C.green : C.red, fontWeight: "700" }}>{snack.text}</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  lightBG: { backgroundColor: "#f7fbff" },
  scroll: { padding: 22, paddingTop: 40, flexGrow: 1 },
  center: { alignItems: "center", marginBottom: 18 },

  logoWrap: {
    width: 112,
    height: 112,
    borderRadius: 24,
    backgroundColor: C.card,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: C.shadow,
    shadowOpacity: 0.2,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 6,
  },
  logoInner: {
    width: 92,
    height: 92,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  title: { color: C.text, fontSize: 30, fontWeight: "900", marginTop: 12 },
  subtitle: { color: C.sub, marginTop: 6, textAlign: "center", maxWidth: 340 },

  rowBetween: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  label: { color: C.text, fontWeight: "700", fontSize: 15 },

  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 10,
    paddingHorizontal: 10,
    paddingVertical: Platform.OS === "ios" ? 12 : 8,
    borderRadius: 12,
    backgroundColor: "#071025",
    borderWidth: 1,
    borderColor: C.border,
  },
  inputFocus: { borderColor: C.primary },
  input: { color: "#fff", fontSize: 15, paddingHorizontal: 6, paddingVertical: 6 },

  controlRow: { flexDirection: "row", alignItems: "center", marginTop: 12 },
  btn: {
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    marginLeft: 10,
    paddingHorizontal: 18,
  },
  ghostBtn: { flex: 0.8, backgroundColor: C.chip, marginLeft: 12 },
  btnText: { color: "#fff", fontWeight: "800" },

  primaryBtn: {
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: C.shadow,
    shadowOpacity: 0.18,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 6 },
    elevation: 4,
  },

  footer: { alignItems: "center", marginTop: 18 },

  snack: {
    position: "absolute",
    left: 16,
    right: 16,
    bottom: 26,
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    alignItems: "center",
    justifyContent: "center",
  },
});
