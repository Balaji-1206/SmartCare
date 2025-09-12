// src/pages/Inventory.tsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  TextInput,
  RefreshControl,
  TouchableOpacity,
  ActivityIndicator,
  Modal,
  Pressable,
  StyleSheet,
  Alert,
  Animated,
  LayoutAnimation,
  Platform,
  UIManager,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Card, Pill } from "../ui";
import { C, getApiBase, API_DEFAULT } from "../constants";
import { apiGet, apiPost, InventoryRow } from "../api";

// enable LayoutAnimation on Android
if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

export default function InventoryScreen({ route }: any) {
  const focusCode: string | undefined = route?.params?.focus;
  const [API_BASE, setAPIBase] = useState<string>(API_DEFAULT);
  const [inv, setInv] = useState<Record<string, InventoryRow>>({});
  const [loading, setLoading] = useState(false);

  // new states
  const [query, setQuery] = useState("");
  const [showLowOnly, setShowLowOnly] = useState(false);
  const [compact, setCompact] = useState(false);

  const [editing, setEditing] = useState<null | { code: string; on_hand: string; reorder_point: string }>(null);
  const [modalVisible, setModalVisible] = useState(false);

  useEffect(() => {
    (async () => setAPIBase(await getApiBase(API_DEFAULT)))();
  }, []);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiGet<Record<string, InventoryRow>>(API_BASE, "/inventory");
      setInv(data || {});
      // animated layout update for lists
     LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);

    } catch (e: any) {
      Alert.alert("SmartCare", e?.message ?? "Failed to load inventory");
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    load();
  }, [load]);

  const startEdit = (code: string, row: InventoryRow) => {
    setEditing({ code, on_hand: String(row.on_hand ?? 0), reorder_point: String(row.reorder_point ?? 0) });
    setModalVisible(true);
  };

  const saveEdit = async () => {
    if (!editing) return;
    const code = editing.code;
    try {
      const on_hand = Number(editing.on_hand || 0);
      const reorder_point = Number(editing.reorder_point || 0);
      await apiPost(API_BASE, "/inventory/upsert", { item_code: code, on_hand, reorder_point });
      setModalVisible(false);
      setEditing(null);
      await load();
    } catch (e: any) {
      Alert.alert("SmartCare", e?.message ?? "Failed to save");
    }
  };

  const stockColor = (row: InventoryRow) => {
    if (row.on_hand <= row.reorder_point) return C.red;
    if (row.on_hand <= row.reorder_point * 1.5) return C.yellow;
    return C.green;
  };

  // Animated progress value store — keep per-item refs in a map
  const animatedMap = useRef<Record<string, Animated.Value>>({}).current;
  const ensureAnimated = (code: string, start = 0) => {
    if (!animatedMap[code]) animatedMap[code] = new Animated.Value(start);
    return animatedMap[code];
  };

  // filtered + sorted list
  const shownItems = useMemo(() => {
    const q = query.trim().toLowerCase();
    return Object.entries(inv)
      .filter(([code, row]) => {
        if (showLowOnly && row.on_hand > row.reorder_point) return false;
        if (!q) return true;
        return row.name.toLowerCase().includes(q) || code.toLowerCase().includes(q);
      })
      .sort((a, b) => a[1].name.localeCompare(b[1].name));
  }, [inv, query, showLowOnly]);

  // when inventory changes, animate progress bars
  useEffect(() => {
    Object.entries(inv).forEach(([code, row]) => {
      const max = Math.max(row.reorder_point * 2, row.on_hand, 10);
      const target = Math.min(1, row.on_hand / max);
      const anim = ensureAnimated(code, 0);
      Animated.timing(anim, { toValue: target, duration: 450, useNativeDriver: false }).start();
    });
  }, [inv]);

  const quickUpdate = async (code: string, patch: Partial<InventoryRow>) => {
    try {
      await apiPost(API_BASE, "/inventory/upsert", { item_code: code, ...patch });
      await load();
    } catch (e: any) {
      Alert.alert("SmartCare", e?.message ?? "Failed");
    }
  };

  // render in compact row or card
  const renderCompact = (code: string, row: InventoryRow) => {
    const anim = ensureAnimated(code);
    return (
      <Pressable key={code} onPress={() => startEdit(code, row)} style={styles.compactRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.itemTitle}>{row.name}</Text>
          <Text style={styles.itemSubSmall}>On hand: {row.on_hand} • Reorder: {row.reorder_point}</Text>
        </View>

        <Pill text={`${row.on_hand}`} bg={stockColor(row)} />
      </Pressable>
    );
  };

  const renderCard = (code: string, row: InventoryRow) => {
    const max = Math.max(row.reorder_point * 2, row.on_hand, 10);
    const pct = Math.min(1, row.on_hand / max);
    const anim = ensureAnimated(code);

    // animated width via interpolated value
    const widthInterp = anim.interpolate({
      inputRange: [0, 1],
      outputRange: ["0%", "100%"],
    });

    return (
      <Card key={code} style={{ marginBottom: 12 }}>
        <View style={styles.rowHeader}>
          <View style={{ flex: 1 }}>
            <Text style={styles.itemTitle}>{row.name}</Text>
            {/* removed item_code display to avoid duplication */}
            <Text style={styles.itemSubSmall}>{/* small subtitle area left intentionally empty */}</Text>
          </View>

          <View style={{ alignItems: "flex-end" }}>
            <Pill text={`${row.on_hand}`} bg={stockColor(row)} />
            <TouchableOpacity onPress={() => startEdit(code, row)} style={{ marginTop: 8 }}>
              <Ionicons name="pencil" size={18} color={C.sub} />
            </TouchableOpacity>
          </View>
        </View>

        <View style={{ marginTop: 12 }}>
          <View style={styles.progressBarBg}>
            <Animated.View style={[styles.progressBarFg, { width: widthInterp, backgroundColor: stockColor(row) }]} />
          </View>
          <View style={styles.progressMeta}>
            <Text style={styles.progressText}>On hand: <Text style={{ fontWeight: "700", color: C.text }}>{row.on_hand}</Text></Text>
            <Text style={styles.progressText}>Reorder at: <Text style={{ fontWeight: "700", color: C.text }}>{row.reorder_point}</Text></Text>
          </View>
        </View>

        <View style={{ flexDirection: "row", marginTop: 12, alignItems: "center" }}>
          <TextInput
            placeholder="Quick set on-hand"
            placeholderTextColor={C.sub}
            keyboardType="number-pad"
            onSubmitEditing={(e) => {
              const val = Number(e.nativeEvent.text || 0);
              quickUpdate(code, { on_hand: val });
            }}
            style={styles.inlineInput}
          />
          <TouchableOpacity
            onPress={() => quickUpdate(code, { on_hand: Math.max(0, row.on_hand - 1) })}
            style={styles.smallBtn}
          >
            <Text style={{ color: "#fff", fontWeight: "700" }}>-1</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => quickUpdate(code, { on_hand: row.on_hand + 1 })}
            style={[styles.smallBtn, { marginLeft: 8 }]}
          >
            <Text style={{ color: "#fff", fontWeight: "700" }}>+1</Text>
          </TouchableOpacity>
        </View>

        {focusCode === code ? <Text style={{ color: C.primary, marginTop: 10 }}>(Opened from Alerts)</Text> : null}
      </Card>
    );
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <ScrollView
        contentContainerStyle={{ padding: 16 }}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={C.primary} />}
      >
        {/* Header */}
        <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 12 }}>
          <Ionicons name="cube-outline" size={24} color={C.primary} style={{ marginRight: 8 }} />
          <Text style={{ color: C.text, fontSize: 22, fontWeight: "800", flex: 1 }}>Inventory</Text>

          <TouchableOpacity onPress={() => { setCompact((s) => !s); }} style={{ padding: 8 }}>
            <Ionicons name={compact ? "list" : "grid-outline"} size={20} color={C.sub} />
          </TouchableOpacity>
        </View>

        {/* Controls: Search + toggles */}
        <View style={{ flexDirection: "row", marginBottom: 12, alignItems: "center" }}>
          <View style={{ flex: 1 }}>
            <TextInput
              placeholder="Search items..."
              placeholderTextColor={C.sub}
              value={query}
              onChangeText={setQuery}
              style={styles.searchInput}
            />
          </View>

          <TouchableOpacity
            onPress={() => setShowLowOnly((s) => !s)}
            style={{
              marginLeft: 10,
              backgroundColor: showLowOnly ? C.primary : C.chip,
              paddingHorizontal: 12,
              paddingVertical: 10,
              borderRadius: 10,
            }}
          >
            <Text style={{ color: showLowOnly ? "#fff" : C.text, fontWeight: "700" }}>{showLowOnly ? "Low only" : "All"}</Text>
          </TouchableOpacity>
        </View>

        {/* Loading */}
        {loading ? (
          <ActivityIndicator color={C.primary} size="large" style={{ marginTop: 40 }} />
        ) : shownItems.length === 0 ? (
          <Card style={{ alignItems: "center", paddingVertical: 40 }}>
            <Text style={{ color: C.sub, textAlign: "center", marginBottom: 8 }}>No matching items</Text>
            <Text style={{ color: C.sub, fontSize: 12 }}>Try clearing search or toggles</Text>
          </Card>
        ) : compact ? (
          // compact mode
          shownItems.map(([code, row]) => renderCompact(code, row))
        ) : (
          // card mode
          shownItems.map(([code, row]) => renderCard(code, row))
        )}

        <View style={{ height: 80 }} />

        {/* Edit Modal */}
        <Modal visible={modalVisible} transparent animationType="fade">
          <View style={styles.modalBackdrop}>
            <View style={styles.modalCard}>
              <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                <Text style={{ fontSize: 18, fontWeight: "800", color: C.text }}>Edit item</Text>
                <Pressable onPress={() => { setModalVisible(false); setEditing(null); }}>
                  <Ionicons name="close" size={22} color={C.sub} />
                </Pressable>
              </View>

              <View style={{ marginTop: 12 }}>
                <Text style={{ color: C.sub, marginBottom: 6 }}>On hand</Text>
                <TextInput
                  value={editing?.on_hand}
                  onChangeText={(v) => setEditing((s) => (s ? { ...s, on_hand: v } : s))}
                  keyboardType="number-pad"
                  style={styles.modalInput}
                />

                <Text style={{ color: C.sub, marginTop: 12, marginBottom: 6 }}>Reorder point</Text>
                <TextInput
                  value={editing?.reorder_point}
                  onChangeText={(v) => setEditing((s) => (s ? { ...s, reorder_point: v } : s))}
                  keyboardType="number-pad"
                  style={styles.modalInput}
                />
              </View>

              <View style={{ flexDirection: "row", marginTop: 16 }}>
                <TouchableOpacity onPress={() => { setModalVisible(false); setEditing(null); }} style={[styles.modalBtn, { backgroundColor: "#e5e7eb" }]}>
                  <Text style={{ fontWeight: "700" }}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={saveEdit} style={[styles.modalBtn, { backgroundColor: C.primary, marginLeft: 10 }]}>
                  <Text style={{ color: "#fff", fontWeight: "800" }}>Save</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  rowHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  itemTitle: { color: C.text, fontSize: 16, fontWeight: "800" },
  itemSubSmall: { color: C.sub, fontSize: 12, marginTop: 4 },
  progressBarBg: { height: 8, backgroundColor: "#f1f5f9", borderRadius: 6, overflow: "hidden" },
  progressBarFg: { height: 8, borderRadius: 6 },
  progressMeta: { flexDirection: "row", justifyContent: "space-between", marginTop: 8 },
  progressText: { color: C.sub, fontSize: 12 },
  inlineInput: {
    flex: 1,
    backgroundColor: "#f3f4f6",
    color: C.text,
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: C.border,
  },
  smallBtn: {
    backgroundColor: C.primary,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 10,
    marginLeft: 8,
    alignItems: "center",
    justifyContent: "center",
  },

  // compact row
  compactRow: { backgroundColor: C.card, padding: 12, borderRadius: 12, marginBottom: 10, flexDirection: "row", alignItems: "center" },

  // modal
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.45)",
    justifyContent: "center",
    padding: 20,
  },
  modalCard: {
    backgroundColor: C.card,
    borderRadius: 14,
    padding: 16,
    shadowColor: C.shadow,
    shadowOpacity: 0.18,
    shadowRadius: 10,
    elevation: 6,
  },
  modalInput: {
    backgroundColor: "#f3f4f6",
    color: C.text,
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: C.border,
  },
  modalBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },

  // search
  searchInput: {
    backgroundColor: "#fff",
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: C.border,
    color: C.text,
  },
});
