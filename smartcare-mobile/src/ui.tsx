// src/ui.tsx
import React from "react";
import { View, Text, TouchableOpacity, ViewStyle, StyleProp } from "react-native";
import { C } from "./constants";

export type CardProps = {
  title?: string;
  children: React.ReactNode;
  accentLeft?: string;
  style?: StyleProp<ViewStyle>;
};

export const Card: React.FC<CardProps> = ({ title, children, accentLeft, style }) => (
  <View
    style={[
      {
        backgroundColor: C.card,
        padding: 16,
        borderRadius: 16,
        marginTop: 14,
        borderLeftWidth: accentLeft ? 6 : 0,
        borderLeftColor: accentLeft || "transparent",
        shadowColor: C.shadow,
        shadowOpacity: 0.15,
        shadowRadius: 6,
        shadowOffset: { width: 0, height: 3 },
        elevation: 3,
      } as ViewStyle,
      style,
    ]}
  >
    {title ? (
      <Text style={{ color: C.text, fontWeight: "700", fontSize: 18, marginBottom: 10 }}>{title}</Text>
    ) : null}
    {children}
  </View>
);

export const Chip: React.FC<{ label: string; bg?: string }> = ({ label, bg }) => (
  <View
    style={{
      backgroundColor: bg || C.chip,
      paddingVertical: 6,
      paddingHorizontal: 12,
      borderRadius: 999,
      marginRight: 8,
      marginBottom: 8,
    }}
  >
    <Text style={{ color: C.text, fontSize: 13, fontWeight: "500" }}>{label}</Text>
  </View>
);

export const Button: React.FC<{ title: string; onPress: () => void; disabled?: boolean }> = ({
  title,
  onPress,
  disabled,
}) => (
  <TouchableOpacity
    disabled={disabled}
    onPress={onPress}
    style={{
      backgroundColor: disabled ? "#93c5fd" : C.primary,
      paddingVertical: 14,
      borderRadius: 12,
      alignItems: "center",
      opacity: disabled ? 0.6 : 1,
      shadowColor: C.shadow,
      shadowOpacity: 0.25,
      shadowRadius: 5,
      shadowOffset: { width: 0, height: 2 },
      elevation: 2,
    }}
  >
    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 16 }}>{title}</Text>
  </TouchableOpacity>
);

export const ErrorBanner: React.FC<{ msg: string | null }> = ({ msg }) =>
  !msg ? null : (
    <View
      style={{
        backgroundColor: "#fee2e2",
        padding: 12,
        borderRadius: 12,
        marginTop: 10,
        borderWidth: 1,
        borderColor: "#fca5a5",
      }}
    >
      <Text style={{ color: "#991b1b", fontWeight: "600" }}>{msg}</Text>
    </View>
  );

export const Pill = ({ text, bg }: { text: string; bg: string }) => (
  <View style={{ backgroundColor: bg, paddingVertical: 5, paddingHorizontal: 12, borderRadius: 999 }}>
    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 12 }}>{text}</Text>
  </View>
);
