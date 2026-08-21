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
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Card, Button, Pill, ErrorBanner } from "../ui";
import {
  C,
  API_DEFAULT,
  getApiBase,
  setApiBase,
  getNurseName,
  setNurseName,
  setAuthed,
} from "../constants";

export default function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [api, setApi] = useState(API_DEFAULT);
  const [name, setName] = useState("");
  const [loadingTest, setLoadingTest] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testStatus, setTestStatus] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [remember, setRemember] = useState<boolean>(true);
  const [apiFocus, setApiFocus] = useState(false);
  const [nameFocus, setNameFocus] = useState(false);

  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.06, duration: 1200, useNativeDriver: false }),
        Animated.timing(pulseAnim, { toValue: 1.0, duration: 1200, useNativeDriver: false }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [pulseAnim]);

  useEffect(() => {
    (async () => {
      try {
        const base = await getApiBase(API_DEFAULT);
        setApi(base);
        const savedName = await getNurseName();
        if (savedName) setName(savedName);
      } catch {}
    })();
  }, []);

  async function testApi() {
    setTestStatus(null);
    setErr(null);
    setLoadingTest(true);
    try {
      const url = api.replace(/\/+$/, "") || API_DEFAULT;
      const t0 = Date.now();
      const r = await fetch(`${url}/`);
      const latency = Date.now() - t0;
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const j = await r.json();
      setTestStatus(`Connected to ${j?.app || "API"} (${latency}ms)`);
    } catch (e: any) {
      const m = e?.message ?? "Cannot reach API server";
      setErr(m);
      setTestStatus(null);
      Alert.alert("Connection Failed", `${m}\n\nMake sure the FastAPI backend is running.`);
    } finally {
      setLoadingTest(false);
    }
  }

  async function onLogin() {
    setErr(null);
    if (!name.trim()) {
      setErr("Please enter a Nurse Display Name");
      return;
    }
    try {
      setSaving(true);
      await setApiBase(api.trim());
      if (remember) {
        await setNurseName(name.trim());
      } else {
        await setNurseName("");
      }
      await setAuthed(true);
      onLoggedIn();
    } catch (e: any) {
      setSaving(false);
      setErr(e?.message ?? "Failed to save login state");
    }
  }

  function fillDemo() {
    setName("Sister Meena (PHC Ward A)");
    setApi(API_DEFAULT);
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <ScrollView
          contentContainerStyle={styles.scroll}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* Glowing Brand Hero */}
          <View style={styles.heroSection}>
            <Animated.View style={[styles.glowRing, { transform: [{ scale: pulseAnim }] }]}>
              <View style={styles.iconCircle}>
                <Ionicons name="pulse" size={42} color={C.primary} />
              </View>
            </Animated.View>

            <Text style={styles.brandTitle}>SmartCare</Text>
            <View style={{ flexDirection: "row", alignItems: "center", marginTop: 6 }}>
              <Pill text="AI Clinical Intelligence" bg={C.primaryBg} textColor={C.primary} />
            </View>
            <Text style={styles.brandSub}>
              Surveillance, Patient Surge Forecasting & PHC Resource Planning
            </Text>
          </View>

          <ErrorBanner msg={err} />

          {/* Configuration Card */}
          <Card title="Server Endpoint" icon="server-outline">
            <Text style={styles.labelSub}>Target FastAPI backend address</Text>
            <View style={[styles.inputRow, apiFocus && styles.inputFocus]}>
              <Ionicons name="link-outline" size={20} color={apiFocus ? C.primary : C.sub} style={{ marginRight: 10 }} />
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
                <Button
                  title={loadingTest ? "Checking..." : "Ping Server"}
                  onPress={testApi}
                  loading={loadingTest}
                  variant="outline"
                  icon="wifi-outline"
                />
              </View>
              <View style={{ flex: 1 }}>
                <Button
                  title="Reset Default"
                  onPress={() => {
                    setApi(API_DEFAULT);
                    setTestStatus(null);
                  }}
                  variant="secondary"
                  icon="refresh-outline"
                />
              </View>
            </View>

            {testStatus && (
              <View style={styles.pingSuccess}>
                <Ionicons name="checkmark-circle" size={16} color={C.green} style={{ marginRight: 6 }} />
                <Text style={{ color: C.green, fontWeight: "700", fontSize: 12 }}>{testStatus}</Text>
              </View>
            )}
          </Card>

          {/* Nurse Identity Card */}
          <Card title="Nurse Station Sign In" icon="person-circle-outline" style={{ marginTop: 14 }}>
            <Text style={styles.labelSub}>Staff name for triage logs and symptom tracking</Text>
            <View style={[styles.inputRow, nameFocus && styles.inputFocus]}>
              <Ionicons name="person-outline" size={20} color={nameFocus ? C.primary : C.sub} style={{ marginRight: 10 }} />
              <TextInput
                value={name}
                onChangeText={setName}
                placeholder="e.g. Meena (Duty Nurse)"
                placeholderTextColor={C.textMuted}
                style={styles.input}
                onFocus={() => setNameFocus(true)}
                onBlur={() => setNameFocus(false)}
              />
            </View>

            <TouchableOpacity
              onPress={() => setRemember((r) => !r)}
              activeOpacity={0.8}
              style={styles.rememberRow}
            >
              <Ionicons
                name={remember ? "checkbox" : "square-outline"}
                size={22}
                color={remember ? C.primary : C.sub}
                style={{ marginRight: 10 }}
              />
              <Text style={{ color: C.textSecondary, fontSize: 13, fontWeight: "600" }}>
                Keep me signed in on this station
              </Text>
            </TouchableOpacity>
          </Card>

          {/* Action Buttons */}
          <View style={{ marginTop: 22, gap: 12 }}>
            <Button
              title={saving ? "Authenticating..." : "Enter Clinical Dashboard"}
              onPress={onLogin}
              loading={saving}
              icon="arrow-forward"
            />
            
            <TouchableOpacity onPress={fillDemo} activeOpacity={0.7} style={styles.demoBtn}>
              <Text style={styles.demoText}>⚡ Quick Fill Sample Profile</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.footer}>
            <Text style={styles.footerText}>SmartCare • Version 1.0.0 • Primary Health Center Suite</Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: C.bg,
  },
  scroll: {
    padding: 20,
    paddingBottom: 40,
  },
  heroSection: {
    alignItems: "center",
    marginTop: 16,
    marginBottom: 10,
  },
  glowRing: {
    width: 86,
    height: 86,
    borderRadius: 43,
    backgroundColor: C.primaryBg,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
    borderWidth: 1,
    borderColor: "rgba(2, 132, 199, 0.25)",
  },
  iconCircle: {
    width: 66,
    height: 66,
    borderRadius: 33,
    backgroundColor: "#ffffff",
    alignItems: "center",
    justifyContent: "center",
    shadowColor: C.shadow,
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  brandTitle: {
    fontSize: 32,
    fontWeight: "900",
    color: C.text,
    letterSpacing: -0.5,
  },
  brandSub: {
    fontSize: 13,
    color: C.sub,
    textAlign: "center",
    marginTop: 8,
    maxWidth: 300,
    lineHeight: 18,
  },
  labelSub: {
    color: C.sub,
    fontSize: 12,
    marginBottom: 8,
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: C.inputBg,
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: Platform.OS === "ios" ? 14 : 10,
    borderWidth: 1,
    borderColor: C.inputBorder,
  },
  inputFocus: {
    borderColor: C.inputFocusBorder,
    backgroundColor: "#ffffff",
  },
  input: {
    flex: 1,
    color: C.text,
    fontSize: 15,
    fontWeight: "600",
  },
  rememberRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 14,
  },
  pingSuccess: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: C.greenBg,
    padding: 10,
    borderRadius: 10,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#bbf7d0",
  },
  demoBtn: {
    alignItems: "center",
    paddingVertical: 10,
  },
  demoText: {
    color: C.primary,
    fontWeight: "700",
    fontSize: 13,
  },
  footer: {
    alignItems: "center",
    marginTop: 28,
  },
  footerText: {
    color: C.textMuted,
    fontSize: 11,
    fontWeight: "500",
  },
});
