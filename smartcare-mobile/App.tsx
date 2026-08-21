// App.tsx
import "react-native-gesture-handler";
import React, { useCallback, useEffect, useState } from "react";
import { View, Text, Platform } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Ionicons } from "@expo/vector-icons";

import { C, isAuthed, getApiBase, API_DEFAULT } from "./src/constants";
import { apiGet } from "./src/api";

import LoginScreen from "./src/pages/Login";
import HomeScreen from "./src/pages/Home";
import AlertsScreen from "./src/pages/Alerts";
import InventoryScreen from "./src/pages/Inventory";
import SettingsScreen from "./src/pages/Settings";

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

// ── Alert badge that polls for count ─────────────────────────
function useLiveAlertCount() {
  const [count, setCount] = useState(0);
  const fetchCount = useCallback(async () => {
    try {
      const base = await getApiBase(API_DEFAULT);
      const res = await apiGet<{ alerts: any[] }>(base, "/alerts");
      setCount((res.alerts || []).filter((a) => a.severity === "HIGH").length);
    } catch {
      // silently fail — badge stays at previous
    }
  }, []);

  useEffect(() => {
    fetchCount();
    const interval = setInterval(fetchCount, 60_000); // refresh every 60s
    return () => clearInterval(interval);
  }, [fetchCount]);

  return count;
}

function MainTabs({ onLogout }: { onLogout: () => void }) {
  const alertCount = useLiveAlertCount();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: "#ffffff",
          borderTopColor: "#e2e8f0",
          borderTopWidth: 1,
          height: Platform.OS === "ios" ? 90 : 66,
          paddingBottom: Platform.OS === "ios" ? 28 : 10,
          paddingTop: 8,
          shadowColor: "#000",
          shadowOpacity: 0.06,
          shadowRadius: 10,
          elevation: 6,
        },
        tabBarActiveTintColor: C.primary,
        tabBarInactiveTintColor: "#94a3b8",
        tabBarLabelStyle: {
          fontWeight: "800",
          fontSize: 11,
          letterSpacing: 0.2,
          marginTop: 2,
        },
        tabBarIcon: ({ color, focused, size }) => {
          const iconMap: Record<string, { active: keyof typeof Ionicons.glyphMap; inactive: keyof typeof Ionicons.glyphMap }> = {
            Today:     { active: "pulse",          inactive: "pulse-outline" },
            Alerts:    { active: "alert-circle",   inactive: "alert-circle-outline" },
            Inventory: { active: "cube",            inactive: "cube-outline" },
            Settings:  { active: "settings",       inactive: "settings-outline" },
          };
          const icons = iconMap[route.name] ?? { active: "ellipse", inactive: "ellipse-outline" };
          const iconName = focused ? icons.active : icons.inactive;

          const isAlerts = route.name === "Alerts";
          const showBadge = isAlerts && alertCount > 0;

          return (
            <View style={{ alignItems: "center", justifyContent: "center" }}>
              <View
                style={
                  focused
                    ? {
                        backgroundColor: C.primaryBg,
                        paddingHorizontal: 14,
                        paddingVertical: 5,
                        borderRadius: 14,
                      }
                    : undefined
                }
              >
                <Ionicons name={iconName} size={focused ? 22 : 21} color={color} />
              </View>
              {showBadge && (
                <View
                  style={{
                    position: "absolute",
                    top: -4,
                    right: -8,
                    backgroundColor: C.red,
                    width: 18,
                    height: 18,
                    borderRadius: 9,
                    alignItems: "center",
                    justifyContent: "center",
                    borderWidth: 2,
                    borderColor: "#ffffff",
                  }}
                >
                  <Text style={{ color: "#fff", fontWeight: "900", fontSize: 9 }}>
                    {alertCount > 9 ? "9+" : alertCount}
                  </Text>
                </View>
              )}
            </View>
          );
        },
      })}
    >
      <Tab.Screen name="Today" component={HomeScreen} />
      <Tab.Screen name="Alerts" component={AlertsScreen} />
      <Tab.Screen name="Inventory" component={InventoryScreen} />
      <Tab.Screen name="Settings">
        {(props) => <SettingsScreen {...props} onLogout={onLogout} />}
      </Tab.Screen>
    </Tab.Navigator>
  );
}

export default function App() {
  const [ready, setReady] = useState(false);
  const [authed, setAuthedState] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const initialAuthed = await isAuthed();
        setAuthedState(initialAuthed);
      } catch {
        setAuthedState(false);
      } finally {
        setReady(true);
      }
    })();
  }, []);

  const handleLoggedIn = () => setAuthedState(true);
  const handleLoggedOut = () => setAuthedState(false);

  if (!ready) {
    return null;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!authed ? (
          <Stack.Screen name="Login">
            {(props) => <LoginScreen {...props} onLoggedIn={handleLoggedIn} />}
          </Stack.Screen>
        ) : (
          <Stack.Screen name="Main">
            {(props) => <MainTabs onLogout={handleLoggedOut} />}
          </Stack.Screen>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
