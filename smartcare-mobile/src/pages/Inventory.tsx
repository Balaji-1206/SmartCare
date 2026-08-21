// src/pages/Inventory.tsx
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  TextInput,
  RefreshControl,
  TouchableOpacity,
  Modal,
  StyleSheet,
  Alert,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Pill, Button, ProgressBar, ErrorBanner, StatCard, LoadingSkeleton, EmptyState, Divider } from "../ui";
import { C, getApiBase, API_DEFAULT, formatLabel } from "../constants";
import { apiGet, InventoryRow, EnrichedInventoryRow } from "../api";
import { onlineOrQueue } from "../offlineQueue";

type InventoryItem = EnrichedInventoryRow & { code: string };

export default function InventoryScreen({ route }: any) {
  const focusCode: string | undefined = route?.params?.focus;
  const [apiBase, setApiBase] = useState(API_DEFAULT);
  const [inventory, setInventory] = useState<Record<string, InventoryRow>>({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState(focusCode || "");
  const [statusFilter, setStatusFilter] = useState<"ALL" | "CRITICAL" | "LOW" | "HEALTHY">("ALL");
  const [err, setErr] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Edit Modal
  const [editingItem, setEditingItem] = useState<{
    code: string; name: string; onHand: string; reorderPoint: string;
  } | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [savingEdit, setSavingEdit] = useState(false);

  // Add New Item Modal
  const [addModalVisible, setAddModalVisible] = useState(false);
  const [newCode, setNewCode] = useState("");
  const [newName, setNewName] = useState("");
  const [newOnHand, setNewOnHand] = useState("0");
  const [newReorder, setNewReorder] = useState("0");
  const [savingNew, setSavingNew] = useState(false);

  useEffect(() => {
    (async () => setApiBase(await getApiBase(API_DEFAULT)))();
  }, []);

  const loadInventory = useCallback(async (isRefresh = false) => {
    try {
      setErr(null);
      if (!isRefresh) setLoading(true);
      const res = await apiGet<Record<string, EnrichedInventoryRow>>(apiBase, "/inventory/enriched");
      setInventory(res || {});
    } catch (e: any) {
      // fallback to plain inventory if enriched is not available
      try {
        const res2 = await apiGet<Record<string, InventoryRow>>(apiBase, "/inventory");
        setInventory(res2 || {});
      } catch {
        setErr(e?.message ?? "Failed to fetch inventory");
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [apiBase]);

  useEffect(() => {
    loadInventory();
  }, [loadInventory]);

  const onRefresh = () => {
    setRefreshing(true);
    loadInventory(true);
  };

  const getStockStatus = (onHand: number, reorderPoint: number) => {
    if (onHand <= 0)                         return { level: "CRITICAL" as const, color: C.red,    bg: C.redBg,    label: "Out of Stock",     urgency: 3 };
    if (onHand <= reorderPoint * 0.5)        return { level: "CRITICAL" as const, color: C.red,    bg: C.redBg,    label: "Critical Shortage", urgency: 2 };
    if (onHand <= reorderPoint)              return { level: "LOW"      as const, color: C.yellow,  bg: C.yellowBg, label: "Reorder Needed",    urgency: 1 };
    return                                          { level: "HEALTHY"  as const, color: C.green,   bg: C.greenBg,  label: "Adequate Stock",    urgency: 0 };
  };

  const getDaysOfSupply = (onHand: number, yhat: number) => {
    if (yhat <= 0) return null;
    return Math.floor(onHand / yhat);
  };

  const itemsList = useMemo<InventoryItem[]>(() => {
    const list = Object.entries(inventory).map(([code, data]) => ({ code, ...data }));
    const q = searchQuery.trim().toLowerCase();

    return list
      .filter((it) => {
        const status = getStockStatus(it.on_hand, it.reorder_point);
        if (statusFilter !== "ALL" && status.level !== statusFilter) return false;
        if (!q) return true;
        return it.code.toLowerCase().includes(q) || (it.name && it.name.toLowerCase().includes(q));
      })
      .sort((a, b) => {
        const sa = getStockStatus(a.on_hand, a.reorder_point);
        const sb = getStockStatus(b.on_hand, b.reorder_point);
        return sb.urgency - sa.urgency;
      });
  }, [inventory, statusFilter, searchQuery]);

  const inventorySummary = useMemo(() => {
    const all = Object.entries(inventory).map(([code, data]) => ({ code, ...data }));
    const critical = all.filter((it) => getStockStatus(it.on_hand, it.reorder_point).level === "CRITICAL").length;
    const low      = all.filter((it) => getStockStatus(it.on_hand, it.reorder_point).level === "LOW").length;
    const healthy  = all.filter((it) => getStockStatus(it.on_hand, it.reorder_point).level === "HEALTHY").length;
    return { total: all.length, critical, low, healthy };
  }, [inventory]);

  function showSuccess(msg: string) {
    setSuccessMsg(msg);
    setTimeout(() => setSuccessMsg(null), 3000);
  }

  const openEditModal = (code: string, row: InventoryRow) => {
    setEditingItem({
      code,
      name: row.name || formatLabel(code),
      onHand: String(row.on_hand ?? 0),
      reorderPoint: String(row.reorder_point ?? 0),
    });
    setModalVisible(true);
  };

  const quickAdjust = async (code: string, delta: number) => {
    const current = inventory[code];
    if (!current) return;
    const newOnHand = Math.max(0, current.on_hand + delta);

    setInventory((prev) => ({ ...prev, [code]: { ...prev[code], on_hand: newOnHand } }));

    try {
      await onlineOrQueue(apiBase, "/inventory/upsert", {
        item_code: code, on_hand: newOnHand, reorder_point: current.reorder_point,
      });
      showSuccess(`${current.name}: updated to ${newOnHand} units`);
    } catch {
      await loadInventory(true);
    }
  };

  const saveModalEdit = async () => {
    if (!editingItem) return;
    try {
      setSavingEdit(true);
      const on_hand = Math.max(0, parseInt(editingItem.onHand, 10) || 0);
      const reorder_point = Math.max(0, parseInt(editingItem.reorderPoint, 10) || 0);

      const res = await onlineOrQueue(apiBase, "/inventory/upsert", {
        item_code: editingItem.code, name: editingItem.name, on_hand, reorder_point,
      });

      setModalVisible(false);
      setEditingItem(null);

      if (res.queued) {
        setInventory((prev) => ({ ...prev, [editingItem.code]: { name: editingItem.name, on_hand, reorder_point } }));
        Alert.alert("Offline Sync", "Inventory update queued locally.");
      } else {
        await loadInventory(true);
        showSuccess(`${editingItem.name} updated successfully`);
      }
    } catch (e: any) {
      Alert.alert("Save Error", e?.message ?? "Failed to update stock");
    } finally {
      setSavingEdit(false);
    }
  };

  const saveNewItem = async () => {
    if (!newCode.trim()) {
      Alert.alert("Validation", "Item code is required.");
      return;
    }
    try {
      setSavingNew(true);
      const on_hand = Math.max(0, parseInt(newOnHand, 10) || 0);
      const reorder_point = Math.max(0, parseInt(newReorder, 10) || 0);

      await onlineOrQueue(apiBase, "/inventory/upsert", {
        item_code: newCode.trim().toLowerCase().replace(/\s+/g, "_"),
        name: newName.trim() || formatLabel(newCode.trim()),
        on_hand,
        reorder_point,
      });

      setAddModalVisible(false);
      setNewCode(""); setNewName(""); setNewOnHand("0"); setNewReorder("0");
      await loadInventory(true);
      showSuccess("New item added to inventory");
    } catch (e: any) {
      Alert.alert("Error", e?.message ?? "Failed to add item");
    } finally {
      setSavingNew(false);
    }
  };

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
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.primary} />}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>Pharmacy & Supplies</Text>
            <Text style={styles.subTitle}>{inventorySummary.total} items tracked</Text>
          </View>
          <TouchableOpacity onPress={() => setAddModalVisible(true)} activeOpacity={0.8} style={styles.addBtn}>
            <Ionicons name="add" size={20} color="#fff" />
          </TouchableOpacity>
        </View>

        <ErrorBanner msg={err} />
        {successMsg && <ErrorBanner msg={successMsg} type="success" />}

        {/* Summary Stats */}
        <View style={styles.statsRow}>
          <StatCard
            label="Critical"
            value={inventorySummary.critical}
            icon="alert-circle"
            iconColor={C.red}
            iconBg={C.redBg}
            style={{ flex: 1, marginRight: 6 }}
          />
          <StatCard
            label="Low Stock"
            value={inventorySummary.low}
            icon="warning"
            iconColor={C.yellow}
            iconBg={C.yellowBg}
            style={{ flex: 1, marginHorizontal: 6 }}
          />
          <StatCard
            label="Healthy"
            value={inventorySummary.healthy}
            icon="checkmark-circle"
            iconColor={C.green}
            iconBg={C.greenBg}
            style={{ flex: 1, marginLeft: 6 }}
          />
        </View>

        {/* Search + Status Filter */}
        <View style={styles.searchBox}>
          <Ionicons name="search" size={18} color={C.sub} style={{ marginRight: 8 }} />
          <TextInput
            value={searchQuery}
            onChangeText={setSearchQuery}
            placeholder="Search drugs by name or code..."
            placeholderTextColor={C.textMuted}
            style={styles.searchInput}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery("")}>
              <Ionicons name="close-circle" size={18} color={C.sub} />
            </TouchableOpacity>
          )}
        </View>

        <View style={styles.filterRow}>
          {(["ALL", "CRITICAL", "LOW", "HEALTHY"] as const).map((f) => (
            <TouchableOpacity
              key={f}
              onPress={() => setStatusFilter(f)}
              activeOpacity={0.8}
              style={[
                styles.filterChip,
                statusFilter === f && {
                  backgroundColor:
                    f === "CRITICAL" ? C.red : f === "LOW" ? C.yellow : f === "HEALTHY" ? C.green : C.primary,
                  borderColor: "transparent",
                },
              ]}
            >
              <Text
                style={[
                  styles.filterChipText,
                  statusFilter === f && { color: "#ffffff", fontWeight: "800" },
                ]}
              >
                {f}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Items List */}
        {itemsList.length === 0 ? (
          <EmptyState
            icon="cube-outline"
            title="No Items Found"
            subtitle="No medicine records match your current filter."
            action={
              <Button title="Clear Filters" variant="outline" size="sm"
                onPress={() => { setSearchQuery(""); setStatusFilter("ALL"); }} />
            }
          />
        ) : (
          itemsList.map((item) => {
            const status = getStockStatus(item.on_hand, item.reorder_point);
            const ratio = item.reorder_point > 0 ? item.on_hand / (item.reorder_point * 2) : 1;

            return (
              <View key={item.code} style={[styles.itemCard, { borderLeftColor: status.color }]}>
                <View style={styles.cardTop}>
                  <View style={{ flex: 1, marginRight: 10 }}>
                    <Text style={styles.itemName}>{item.name || formatLabel(item.code)}</Text>
                    <Text style={styles.itemCode}>{item.code}</Text>
                  </View>
                  <Pill text={status.label} bg={status.bg} textColor={status.color} dot size="sm" />
                </View>

                <View style={styles.metricRow}>
                  <View>
                    <Text style={styles.metricLabel}>On Hand</Text>
                    <View style={{ flexDirection: "row", alignItems: "baseline" }}>
                      <Text style={[styles.metricValue, { color: status.color }]}>{item.on_hand}</Text>
                      <Text style={styles.metricUnit}> units</Text>
                    </View>
                  </View>
                  <View style={{ alignItems: "flex-end", gap: 4 }}>
                    <Text style={styles.metricLabel}>Reorder At</Text>
                    <Text style={styles.reorderText}>{item.reorder_point} units</Text>
                    {(item as EnrichedInventoryRow).days_to_stockout != null && (
                      <View style={[
                        styles.dosBadge,
                        { backgroundColor: (item as EnrichedInventoryRow).days_to_stockout! < 7
                          ? C.redBg : (item as EnrichedInventoryRow).days_to_stockout! < 14
                          ? C.yellowBg : C.greenBg }
                      ]}>
                        <Text style={[styles.dosText, {
                          color: (item as EnrichedInventoryRow).days_to_stockout! < 7
                            ? C.red : (item as EnrichedInventoryRow).days_to_stockout! < 14
                            ? C.yellow : C.green
                        }]}>
                          ~{(item as EnrichedInventoryRow).days_to_stockout}d supply
                        </Text>
                      </View>
                    )}
                  </View>
                </View>

                <ProgressBar progress={ratio} color={status.color} height={6} style={{ marginVertical: 10 }} />

                <View style={styles.cardFooter}>
                  {/* Quick adjust steppers */}
                  <View style={styles.stepperRow}>
                    <TouchableOpacity onPress={() => quickAdjust(item.code, -10)} activeOpacity={0.7} style={styles.stepBtn}>
                      <Text style={styles.stepBtnText}>−10</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => quickAdjust(item.code, -1)} activeOpacity={0.7} style={styles.stepBtn}>
                      <Text style={styles.stepBtnText}>−1</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => quickAdjust(item.code, 1)} activeOpacity={0.7} style={styles.stepBtn}>
                      <Text style={styles.stepBtnText}>+1</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => quickAdjust(item.code, 10)} activeOpacity={0.7} style={styles.stepBtn}>
                      <Text style={styles.stepBtnText}>+10</Text>
                    </TouchableOpacity>
                  </View>
                  <TouchableOpacity onPress={() => openEditModal(item.code, item)} activeOpacity={0.8} style={styles.editBtn}>
                    <Ionicons name="create-outline" size={14} color={C.primary} style={{ marginRight: 4 }} />
                    <Text style={styles.editBtnText}>Edit</Text>
                  </TouchableOpacity>
                </View>
              </View>
            );
          })
        )}

        {/* Edit Stock Modal */}
        <Modal visible={modalVisible} transparent animationType="fade">
          <View style={styles.modalBackdrop}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Edit Stock Levels</Text>
                <TouchableOpacity onPress={() => setModalVisible(false)}>
                  <Ionicons name="close" size={22} color={C.sub} />
                </TouchableOpacity>
              </View>
              {editingItem && (
                <>
                  <Divider />
                  <Text style={styles.modalDrugName}>{editingItem.name}</Text>
                  <Text style={styles.modalDrugCode}>Code: {editingItem.code}</Text>

                  <Text style={[styles.modalLabel, { marginTop: 16 }]}>Units Currently On Hand</Text>
                  <TextInput value={editingItem.onHand}
                    onChangeText={(v) => setEditingItem((p) => p ? { ...p, onHand: v } : null)}
                    keyboardType="number-pad" style={styles.modalInput} />

                  <Text style={[styles.modalLabel, { marginTop: 12 }]}>Safety Reorder Threshold</Text>
                  <TextInput value={editingItem.reorderPoint}
                    onChangeText={(v) => setEditingItem((p) => p ? { ...p, reorderPoint: v } : null)}
                    keyboardType="number-pad" style={styles.modalInput} />

                  <View style={{ flexDirection: "row", gap: 10, marginTop: 20 }}>
                    <View style={{ flex: 1 }}>
                      <Button title="Cancel" variant="secondary" onPress={() => setModalVisible(false)} />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Button title={savingEdit ? "Saving..." : "Save Changes"} loading={savingEdit} onPress={saveModalEdit} />
                    </View>
                  </View>
                </>
              )}
            </View>
          </View>
        </Modal>

        {/* Add New Item Modal */}
        <Modal visible={addModalVisible} transparent animationType="fade">
          <View style={styles.modalBackdrop}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Add New Medicine</Text>
                <TouchableOpacity onPress={() => setAddModalVisible(false)}>
                  <Ionicons name="close" size={22} color={C.sub} />
                </TouchableOpacity>
              </View>
              <Divider />

              <Text style={styles.modalLabel}>Item Code (unique identifier)</Text>
              <TextInput value={newCode} onChangeText={setNewCode}
                placeholder="e.g. ors_packets" placeholderTextColor={C.textMuted}
                autoCapitalize="none" style={styles.modalInput} />

              <Text style={[styles.modalLabel, { marginTop: 12 }]}>Display Name</Text>
              <TextInput value={newName} onChangeText={setNewName}
                placeholder="e.g. ORS Sachets" placeholderTextColor={C.textMuted}
                style={styles.modalInput} />

              <View style={{ flexDirection: "row", gap: 12, marginTop: 12 }}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.modalLabel}>Units On Hand</Text>
                  <TextInput value={newOnHand} onChangeText={setNewOnHand}
                    keyboardType="number-pad" style={styles.modalInput} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.modalLabel}>Reorder At</Text>
                  <TextInput value={newReorder} onChangeText={setNewReorder}
                    keyboardType="number-pad" style={styles.modalInput} />
                </View>
              </View>

              <View style={{ flexDirection: "row", gap: 10, marginTop: 20 }}>
                <View style={{ flex: 1 }}>
                  <Button title="Cancel" variant="secondary" onPress={() => setAddModalVisible(false)} />
                </View>
                <View style={{ flex: 1 }}>
                  <Button title={savingNew ? "Adding..." : "Add Item"} loading={savingNew} onPress={saveNewItem} icon="add-circle-outline" />
                </View>
              </View>
            </View>
          </View>
        </Modal>
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
  addBtn: {
    width: 40, height: 40, borderRadius: 13, backgroundColor: C.primary,
    alignItems: "center", justifyContent: "center",
    shadowColor: C.primary, shadowOpacity: 0.3, shadowRadius: 6, elevation: 3,
  },
  statsRow: { flexDirection: "row", marginTop: 2, marginBottom: 14 },
  searchBox: {
    flexDirection: "row", alignItems: "center", backgroundColor: "#ffffff",
    borderRadius: 14, paddingHorizontal: 14, paddingVertical: 10,
    borderWidth: 1, borderColor: C.inputBorder,
    shadowColor: "#000", shadowOpacity: 0.04, shadowRadius: 4, elevation: 1,
  },
  searchInput: { flex: 1, color: C.text, fontSize: 14 },
  filterRow: { flexDirection: "row", gap: 8, marginVertical: 10, flexWrap: "wrap" },
  filterChip: {
    paddingVertical: 7, paddingHorizontal: 14, borderRadius: 12,
    backgroundColor: "#ffffff", borderWidth: 1, borderColor: C.border,
  },
  filterChipText: { color: C.textSecondary, fontSize: 12, fontWeight: "600" },
  itemCard: {
    backgroundColor: "#ffffff", borderRadius: 20, padding: 16, marginBottom: 12,
    borderWidth: 1, borderColor: C.border, borderLeftWidth: 5,
    shadowColor: "#000", shadowOpacity: 0.06, shadowRadius: 8, elevation: 2,
  },
  cardTop: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" },
  itemName: { color: C.text, fontSize: 16, fontWeight: "800", flex: 1 },
  itemCode: { color: C.sub, fontSize: 12, marginTop: 2, fontFamily: "monospace" },
  metricRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-end", marginTop: 12 },
  metricLabel: { color: C.sub, fontSize: 11, fontWeight: "600", marginBottom: 2 },
  metricValue: { fontSize: 28, fontWeight: "900" },
  metricUnit: { fontSize: 13, color: C.sub, fontWeight: "600" },
  reorderText: { color: C.textSecondary, fontSize: 15, fontWeight: "700" },
  cardFooter: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  stepperRow: { flexDirection: "row", gap: 6 },
  stepBtn: {
    paddingVertical: 6, paddingHorizontal: 10, backgroundColor: "#f1f5f9",
    borderRadius: 9, borderWidth: 1, borderColor: C.border,
  },
  stepBtnText: { color: C.text, fontSize: 12, fontWeight: "800" },
  editBtn: {
    flexDirection: "row", alignItems: "center",
    backgroundColor: C.primaryBg, paddingVertical: 6, paddingHorizontal: 12, borderRadius: 10,
  },
  editBtnText: { color: C.primary, fontSize: 13, fontWeight: "800" },
  modalBackdrop: {
    flex: 1, backgroundColor: "rgba(15, 23, 42, 0.6)",
    justifyContent: "center", alignItems: "center", padding: 20,
  },
  modalContent: {
    width: "100%", maxWidth: 440, backgroundColor: "#ffffff",
    borderRadius: 24, padding: 20, borderWidth: 1, borderColor: C.border,
    shadowColor: "#000", shadowOpacity: 0.2, shadowRadius: 20, elevation: 8,
  },
  modalHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  modalTitle: { color: C.text, fontSize: 18, fontWeight: "900" },
  modalDrugName: { color: C.primary, fontSize: 16, fontWeight: "800", marginTop: 8 },
  modalDrugCode: { color: C.sub, fontSize: 12, marginTop: 2 },
  modalLabel: { color: C.sub, fontSize: 12, fontWeight: "700", marginBottom: 6 },
  modalInput: {
    backgroundColor: "#f8fafc", borderRadius: 14, padding: 12,
    color: C.text, fontSize: 15, fontWeight: "700", borderWidth: 1, borderColor: C.border,
  },
  dosBadge: {
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8,
  },
  dosText: { fontSize: 11, fontWeight: "800" },
});
