import React, { useEffect, useState } from "react";
import { SafeAreaView, View, Text, TextInput, Pressable } from "react-native";
import { getApiBase, setApiBase, getNurseName, setNurseName } from "../storage";

const C = { bg:"#0b1220", card:"#111827", text:"#fff", sub:"#9ca3af", btn:"#2563eb" };

export default function SettingsScreen() {
  const [api, setApi] = useState("");
  const [nurse, setNurse] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => { (async () => {
    setApi(await getApiBase("http://127.0.0.1:8000"));
    setNurse(await getNurseName());
  })(); }, []);

  async function saveAll() {
    await setApiBase(api);
    await setNurseName(nurse || "Nurse");
    setMsg("Saved");
    setTimeout(()=>setMsg(null), 1200);
  }
  async function test() {
    try {
      const r = await fetch(`${api}/`); 
      const j = await r.json();
      setMsg(`OK: ${j?.app ?? "server"}`);
    } catch(e:any) {
      setMsg("Error reaching API");
    }
  }

  return (
    <SafeAreaView style={{ flex:1, backgroundColor:C.bg }}>
      <View style={{ padding:16 }}>
        <Text style={{ color:C.text, fontSize:22, fontWeight:"800" }}>SmartCare — Settings</Text>

        <View style={{ backgroundColor:C.card, padding:16, borderRadius:12, marginTop:12 }}>
          <Text style={{ color:C.sub, marginBottom:6 }}>API Base URL</Text>
          <TextInput value={api} onChangeText={setApi} autoCapitalize="none"
            placeholder="http://10.0.2.2:8000" placeholderTextColor={C.sub}
            style={{ backgroundColor:C.bg, color:C.text, padding:10, borderRadius:10 }} />
        </View>

        <View style={{ backgroundColor:C.card, padding:16, borderRadius:12, marginTop:12 }}>
          <Text style={{ color:C.sub, marginBottom:6 }}>Nurse Name</Text>
          <TextInput value={nurse} onChangeText={setNurse}
            placeholder="Meena" placeholderTextColor={C.sub}
            style={{ backgroundColor:C.bg, color:C.text, padding:10, borderRadius:10 }} />
        </View>

        <View style={{ flexDirection:"row", gap:12, marginTop:14 }}>
          <Pressable onPress={saveAll} style={{ backgroundColor:C.btn, padding:12, borderRadius:10 }}>
            <Text style={{ color:"#fff", fontWeight:"700" }}>Save</Text>
          </Pressable>
          <Pressable onPress={test} style={{ backgroundColor:C.btn, padding:12, borderRadius:10 }}>
            <Text style={{ color:"#fff", fontWeight:"700" }}>Test API</Text>
          </Pressable>
        </View>
        {msg && <Text style={{ color:C.sub, marginTop:8 }}>{msg}</Text>}
      </View>
    </SafeAreaView>
  );
}
