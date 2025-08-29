import React, { useCallback, useEffect, useMemo, useState } from "react";
import { SafeAreaView, View, Text, RefreshControl, ScrollView, Pressable } from "react-native";
import { getApiBase } from "../storage";

type AlertT = { type:"stockout_risk"|"reorder"; severity:"HIGH"|"MEDIUM"|"LOW"; message:string; item_code:string };
type MobileToday = { critical_alerts: AlertT[] };

const C = { bg:"#0b1220", card:"#111827", text:"#fff", sub:"#9ca3af", chip:"#1f2937", border:"#374151" };

export default function AlertsScreen({ navigation }: any) {
  const [api, setApi] = useState("http://127.0.0.1:8000");
  const [alerts, setAlerts] = useState<AlertT[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<"ALL"|"HIGH"|"MEDIUM"|"LOW">("ALL");

  useEffect(() => { (async () => setApi(await getApiBase(api)))(); }, []);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const r = await fetch(`${api}/mobile/today`);
      const j: MobileToday = await r.json();
      setAlerts(j.critical_alerts || []);
    } finally { setLoading(false); }
  }, [api]);

  useEffect(() => { load(); }, [load]);

  const shown = useMemo(() => filter==="ALL" ? alerts : alerts.filter(a => a.severity===filter), [alerts, filter]);

  return (
    <SafeAreaView style={{ flex:1, backgroundColor:C.bg }}>
      <ScrollView contentContainerStyle={{ padding:16 }} refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor="#fff" />}>
        <Text style={{ color:C.text, fontSize:22, fontWeight:"800" }}>SmartCare — Alerts</Text>

        <View style={{ flexDirection:"row", gap:8, marginTop:12 }}>
          {(["ALL","HIGH","MEDIUM","LOW"] as const).map(f => (
            <Pressable key={f} onPress={() => setFilter(f)} style={{ backgroundColor: filter===f ? "#2563eb" : C.chip, paddingVertical:6, paddingHorizontal:10, borderRadius:999 }}>
              <Text style={{ color:"#fff" }}>{f}</Text>
            </Pressable>
          ))}
        </View>

        <View style={{ marginTop:12 }}>
          {shown.length===0 ? <Text style={{ color:C.sub }}>No alerts</Text> : shown.map((a, i) => (
            <View key={i} style={{ backgroundColor:C.card, padding:12, borderRadius:12, marginBottom:8, borderWidth:1, borderColor:C.border }}>
              <Text style={{ color:"#fff", fontWeight:"700" }}>{a.severity}</Text>
              <Text style={{ color:"#fff", marginTop:4 }}>{a.message}</Text>
              <Pressable onPress={() => navigation.navigate("Inventory", { focus: a.item_code })} style={{ marginTop:8 }}>
                <Text style={{ color:"#60a5fa" }}>View “{a.item_code}” in Inventory →</Text>
              </Pressable>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
