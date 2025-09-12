// App.tsx
import "react-native-gesture-handler";
import React, { useEffect, useState } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Ionicons } from "@expo/vector-icons";

import Login from "./src/pages/Login";
import Home from "./src/pages/Home";
import Alerts from "./src/pages/Alerts";
import Inventory from "./src/pages/Inventory";
import Settings from "./src/pages/Settings";




const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function MainTabs({ onLogout }: { onLogout: () => void }) {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: { backgroundColor: "#fff", borderTopColor: "#e6eef6" },
        tabBarActiveTintColor: "#229ED9",
        tabBarInactiveTintColor: "#6b7280",
        tabBarIcon: ({ color, size }) => {
          const map: Record<string, string> = {
            Today: "pulse-outline",
            Alerts: "alert-circle-outline",
            Inventory: "cube-outline",
            Settings: "settings-outline",
          };
          const name = map[route.name] ?? "ellipse-outline";
          return <Ionicons name={name as any} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Today" component={Home} />
      <Tab.Screen name="Alerts" component={Alerts} />
    
       <Tab.Screen name="Inventory" component={Inventory} /> 
             <Tab.Screen name="Settings" component={() => <Settings onLogout={onLogout} />} />
    </Tab.Navigator>
  );
}

export default function App() {
  const [ready, setReady] = useState(false);
  const [authed, setAuthed] = useState(false);

  useEffect(() => {
    (async () => {
      // keep existing auth check logic inside Login page; simple placeholder here
      // import isAuthed in future to set initial auth state
      setReady(true);
    })();
  }, []);

  if (!ready) return null;

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!authed ? (
          <Stack.Screen name="Login">
            {() => <Login onLoggedIn={() => setAuthed(true)} />}
          </Stack.Screen>
        ) : (
          <Stack.Screen name="Main">
            {() => <MainTabs onLogout={() => setAuthed(false)} />}
          </Stack.Screen>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
