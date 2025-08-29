import React, { useEffect, useState } from "react";
import { SafeAreaView, View, Text, TextInput, Pressable, Alert, Platform } from "react-native";
import { getApiBase, setApiBase, getNurseName, setNurseName, getPin, setPin, setAuthed } from "../storage";

const C = { bg:"#0b1220", card:"#111827", text:"#fff", sub:"#9ca3af", btn:"#2563eb" };

const API_DEFAULT = Platform.select({
  android: "http://10.0.2.2:8000",
  ios: "http://127.0.0.1:8000",
  default: "http://127.0.0.1:8000",
})!;

export default function LoginScreen() {
  const [api, setApi] = useState(API_DEFAULT);
  const [name, setName] = useState("");
  const [pin, setPinLocal] = useState("");

  useEffect(() => {
    (async () => {
      setApi(await getApiBase(API_DEFAULT));
      setName(await getNurseName());
      setPinLocal(await getPin());
    })();
  }, []);

  async function testApi() {
    try {
      const r = await fetch(`${api}/`);
      const j = await r.json();
      Alert.alert("SmartCare", `OK: ${j?.app ?? "API online"}`);
    } catch {
      Alert.alert("SmartCare", "API not reachable. Check URL & server.");
    }
  }

  async function onLogin() {
    if (!name.trim()) { Alert.alert("SmartCare", "Please enter your name"); return; }
    if (pin && pin.length < 4) { Alert.alert("SmartCare", "PIN must be at least 4 digits"); return; }

    await setApiBase(api);
    await setNurseName(name.trim());
    await setPin(pin);
    await setAuthed(true);
    // The root navigator will detect auth state and show the main tabs.
  }

  return (
    <SafeAreaView style={{ flex:1, backgroundColor: C.bg }}>
      <View style={{ padding: 20 }}>
        <Text style={{ color: C.text, fontSize: 28, fontWeight: "800" }}>SmartCare</Text>
        <Text style={{ color: C.sub, marginTop: 4 }}>Sign in to continue</Text>

        <View style={{ backgroundColor: C.card, padding: 16, borderRadius: 12, marginTop: 16 }}>
          <Text style={{ color: C.sub, marginBottom: 6 }}>API Base URL</Text>
          <TextInput
            value={api}
            onChangeText={setApi}
            autoCapitalize="none"
            placeholder={API_DEFAULT}
            placeholderTextColor={C.sub}
            style={{ backgroundColor: C.bg, color: C.text, padding: 12, borderRadius: 10 }}
          />
          <Pressable onPress={testApi} style={{ marginTop: 10, backgroundColor: C.btn, padding: 12, borderRadius: 10 }}>
            <Text style={{ color: "#fff", fontWeight: "700", textAlign: "center" }}>Test API</Text>
          </Pressable>
        </View>

        <View style={{ backgroundColor: C.card, padding: 16, borderRadius: 12, marginTop: 12 }}>
          <Text style={{ color: C.sub, marginBottom: 6 }}>Nurse Name</Text>
          <TextInput
            value={name}
            onChangeText={setName}
            placeholder="Meena"
            placeholderTextColor={C.sub}
            style={{ backgroundColor: C.bg, color: C.text, padding: 12, borderRadius: 10 }}
          />

          <Text style={{ color: C.sub, marginTop: 12, marginBottom: 6 }}>PIN (optional)</Text>
          <TextInput
            value={pin}
            onChangeText={setPinLocal}
            placeholder="4+ digits"
            placeholderTextColor={C.sub}
            keyboardType="number-pad"
            secureTextEntry
            style={{ backgroundColor: C.bg, color: C.text, padding: 12, borderRadius: 10 }}
          />
        </View>

        <Pressable onPress={onLogin} style={{ marginTop: 16, backgroundColor: C.btn, padding: 14, borderRadius: 10 }}>
          <Text style={{ color: "#fff", fontWeight: "700", textAlign: "center" }}>Login</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}
