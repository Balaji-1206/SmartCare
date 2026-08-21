// src/ui.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ViewStyle,
  StyleProp,
  ActivityIndicator,
  Animated,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { C } from "./constants";

// ─── Card ────────────────────────────────────────────────────
export type CardProps = {
  title?: string;
  subtitle?: string;
  icon?: keyof typeof Ionicons.glyphMap;
  iconColor?: string;
  iconBg?: string;
  children: React.ReactNode;
  accentLeft?: string;
  style?: StyleProp<ViewStyle>;
  headerRight?: React.ReactNode;
  onPress?: () => void;
};

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  icon,
  iconColor,
  iconBg,
  children,
  accentLeft,
  style,
  headerRight,
  onPress,
}) => {
  const inner = (
    <View
      style={[
        {
          backgroundColor: C.card,
          padding: 18,
          borderRadius: 20,
          marginTop: 14,
          borderWidth: 1,
          borderColor: C.glassBorder,
          borderLeftWidth: accentLeft ? 5 : 1,
          borderLeftColor: accentLeft || C.glassBorder,
          shadowColor: "#000",
          shadowOpacity: 0.06,
          shadowRadius: 8,
          shadowOffset: { width: 0, height: 3 },
          elevation: 2,
        } as ViewStyle,
        style,
      ]}
    >
      {(title || headerRight) ? (
        <View
          style={{
            flexDirection: "row",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: subtitle ? 2 : 12,
          }}
        >
          <View style={{ flexDirection: "row", alignItems: "center", flex: 1 }}>
            {icon ? (
              <View
                style={{
                  width: 34,
                  height: 34,
                  borderRadius: 10,
                  backgroundColor: iconBg || C.primaryBg,
                  alignItems: "center",
                  justifyContent: "center",
                  marginRight: 10,
                }}
              >
                <Ionicons name={icon} size={18} color={iconColor || C.primary} />
              </View>
            ) : null}
            <View style={{ flex: 1 }}>
              <Text style={{ color: C.text, fontWeight: "800", fontSize: 17, letterSpacing: -0.2 }}>
                {title}
              </Text>
              {subtitle ? (
                <Text style={{ color: C.sub, fontSize: 12, marginTop: 2 }}>{subtitle}</Text>
              ) : null}
            </View>
          </View>
          {headerRight}
        </View>
      ) : null}
      {children}
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.85}>
        {inner}
      </TouchableOpacity>
    );
  }
  return inner;
};

// ─── StatCard ────────────────────────────────────────────────
export const StatCard: React.FC<{
  label: string;
  value: string | number;
  unit?: string;
  icon?: keyof typeof Ionicons.glyphMap;
  iconColor?: string;
  iconBg?: string;
  trend?: number | null;
  style?: StyleProp<ViewStyle>;
}> = ({ label, value, unit, icon, iconColor = C.primary, iconBg = C.primaryBg, trend, style }) => (
  <View
    style={[
      {
        backgroundColor: C.card,
        borderRadius: 18,
        padding: 14,
        flex: 1,
        borderWidth: 1,
        borderColor: C.glassBorder,
        shadowColor: "#000",
        shadowOpacity: 0.05,
        shadowRadius: 6,
        elevation: 2,
      } as ViewStyle,
      style,
    ]}
  >
    <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" }}>
      {icon && (
        <View
          style={{
            width: 32,
            height: 32,
            borderRadius: 10,
            backgroundColor: iconBg,
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Ionicons name={icon} size={16} color={iconColor} />
        </View>
      )}
      {trend != null && (
        <View
          style={{
            backgroundColor: trend >= 0 ? C.greenBg : C.redBg,
            borderRadius: 8,
            paddingHorizontal: 6,
            paddingVertical: 2,
          }}
        >
          <Text
            style={{
              color: trend >= 0 ? C.green : C.red,
              fontSize: 11,
              fontWeight: "800",
            }}
          >
            {trend >= 0 ? "↑" : "↓"} {Math.abs(trend).toFixed(1)}%
          </Text>
        </View>
      )}
    </View>
    <Text style={{ color: C.text, fontSize: 22, fontWeight: "900", marginTop: 8, letterSpacing: -0.5 }}>
      {value}
      {unit && <Text style={{ fontSize: 13, fontWeight: "600", color: C.sub }}> {unit}</Text>}
    </Text>
    <Text style={{ color: C.sub, fontSize: 12, fontWeight: "600", marginTop: 2 }}>{label}</Text>
  </View>
);

// ─── SectionTitle ─────────────────────────────────────────────
export const SectionTitle: React.FC<{
  title: string;
  icon?: keyof typeof Ionicons.glyphMap;
  rightAction?: React.ReactNode;
}> = ({ title, icon, rightAction }) => (
  <View
    style={{
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "center",
      marginTop: 22,
      marginBottom: 6,
      paddingHorizontal: 2,
    }}
  >
    <View style={{ flexDirection: "row", alignItems: "center" }}>
      {icon && <Ionicons name={icon} size={17} color={C.primary} style={{ marginRight: 6 }} />}
      <Text style={{ color: C.text, fontSize: 16, fontWeight: "800", letterSpacing: -0.2 }}>
        {title}
      </Text>
    </View>
    {rightAction}
  </View>
);

// ─── Divider ──────────────────────────────────────────────────
export const Divider: React.FC<{ style?: StyleProp<ViewStyle> }> = ({ style }) => (
  <View
    style={[
      { height: 1, backgroundColor: C.divider, marginVertical: 10 },
      style,
    ]}
  />
);

// ─── Pill ─────────────────────────────────────────────────────
export const Pill: React.FC<{
  text: string;
  bg?: string;
  textColor?: string;
  icon?: keyof typeof Ionicons.glyphMap;
  dot?: boolean;
  size?: "sm" | "md";
}> = ({ text, bg = C.primaryBg, textColor = C.primary, icon, dot, size = "md" }) => {
  return (
    <View
      style={{
        backgroundColor: bg,
        paddingVertical: size === "sm" ? 3 : 5,
        paddingHorizontal: size === "sm" ? 8 : 12,
        borderRadius: 999,
        flexDirection: "row",
        alignItems: "center",
      }}
    >
      {dot && (
        <View
          style={{
            width: 6,
            height: 6,
            borderRadius: 3,
            backgroundColor: textColor,
            marginRight: 5,
          }}
        />
      )}
      {icon && <Ionicons name={icon} size={size === "sm" ? 11 : 13} color={textColor} style={{ marginRight: 4 }} />}
      <Text
        style={{
          color: textColor,
          fontWeight: "800",
          fontSize: size === "sm" ? 10 : 12,
          letterSpacing: 0.3,
          textTransform: "uppercase",
        }}
      >
        {text}
      </Text>
    </View>
  );
};

// ─── Chip ─────────────────────────────────────────────────────
export const Chip: React.FC<{
  label: string;
  active?: boolean;
  onPress?: () => void;
  count?: number;
}> = ({ label, active, onPress, count }) => (
  <TouchableOpacity
    onPress={onPress}
    activeOpacity={0.8}
    style={{
      backgroundColor: active ? C.primary : "#ffffff",
      paddingVertical: 8,
      paddingHorizontal: 14,
      borderRadius: 14,
      marginRight: 8,
      marginBottom: 8,
      flexDirection: "row",
      alignItems: "center",
      borderWidth: 1,
      borderColor: active ? C.primary : C.border,
      shadowColor: "#000",
      shadowOpacity: active ? 0.12 : 0.04,
      shadowRadius: 4,
      elevation: 1,
    }}
  >
    <Text
      style={{
        color: active ? "#ffffff" : C.textSecondary,
        fontSize: 13,
        fontWeight: active ? "800" : "600",
      }}
    >
      {label}
    </Text>
    {count != null && count > 0 && (
      <View
        style={{
          marginLeft: 6,
          backgroundColor: active ? "rgba(255,255,255,0.25)" : C.primaryBg,
          paddingHorizontal: 6,
          paddingVertical: 2,
          borderRadius: 8,
          minWidth: 18,
          alignItems: "center",
        }}
      >
        <Text style={{ color: active ? "#fff" : C.primary, fontSize: 11, fontWeight: "800" }}>
          {count}
        </Text>
      </View>
    )}
  </TouchableOpacity>
);

// ─── Button ───────────────────────────────────────────────────
export const Button: React.FC<{
  title: string;
  onPress: () => void;
  loading?: boolean;
  disabled?: boolean;
  icon?: keyof typeof Ionicons.glyphMap;
  variant?: "primary" | "secondary" | "outline" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  style?: StyleProp<ViewStyle>;
}> = ({ title, onPress, loading, disabled, icon, variant = "primary", size = "md", style }) => {
  const scale = useRef(new Animated.Value(1)).current;

  const onPressIn = () =>
    Animated.spring(scale, { toValue: 0.97, useNativeDriver: false, speed: 40 }).start();
  const onPressOut = () =>
    Animated.spring(scale, { toValue: 1, friction: 5, useNativeDriver: false }).start();

  const sizeMap = {
    sm: { py: 9, px: 14, fontSize: 13 },
    md: { py: 13, px: 18, fontSize: 15 },
    lg: { py: 16, px: 22, fontSize: 16 },
  };
  const s = sizeMap[size];

  let bg = C.primary;
  let textCol = "#ffffff";
  let borderWidth = 0;
  let borderCol = "transparent";

  if (variant === "secondary") { bg = "#f1f5f9"; textCol = C.text; borderWidth = 1; borderCol = C.border; }
  else if (variant === "outline") { bg = "transparent"; textCol = C.primary; borderWidth = 1.5; borderCol = C.primary; }
  else if (variant === "danger") { bg = C.red; textCol = "#ffffff"; }
  else if (variant === "ghost") { bg = "transparent"; textCol = C.primary; }

  if (disabled) { bg = "#e2e8f0"; textCol = C.textMuted; borderWidth = 0; }

  return (
    <Animated.View style={[{ transform: [{ scale }] }, style]}>
      <TouchableOpacity
        disabled={disabled || loading}
        onPress={onPress}
        onPressIn={onPressIn}
        onPressOut={onPressOut}
        activeOpacity={0.9}
        style={{
          backgroundColor: bg,
          paddingVertical: s.py,
          paddingHorizontal: s.px,
          borderRadius: 14,
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "row",
          borderWidth,
          borderColor: borderCol,
          shadowColor: variant === "primary" ? C.primary : "#000",
          shadowOpacity: variant === "primary" ? 0.18 : 0.04,
          shadowRadius: 6,
          shadowOffset: { width: 0, height: 3 },
          elevation: variant === "primary" ? 3 : 1,
        }}
      >
        {loading ? (
          <ActivityIndicator color={textCol} size="small" />
        ) : (
          <>
            {icon && <Ionicons name={icon} size={s.fontSize} color={textCol} style={{ marginRight: 7 }} />}
            <Text style={{ color: textCol, fontWeight: "800", fontSize: s.fontSize, letterSpacing: 0.1 }}>
              {title}
            </Text>
          </>
        )}
      </TouchableOpacity>
    </Animated.View>
  );
};

// ─── StepperInput ─────────────────────────────────────────────
export const StepperInput: React.FC<{
  label: string;
  sublabel?: string;
  value: number;
  onChange: (val: number) => void;
  icon?: keyof typeof Ionicons.glyphMap;
  color?: string;
  max?: number;
}> = ({ label, sublabel, value, onChange, icon, color = C.primary, max = 999 }) => {
  const bgColor = `${color}15`;
  return (
    <View
      style={{
        backgroundColor: value > 0 ? bgColor : C.bgSubtle,
        padding: 12,
        borderRadius: 16,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 10,
        borderWidth: 1,
        borderColor: value > 0 ? color + "40" : C.border,
      }}
    >
      <View style={{ flexDirection: "row", alignItems: "center", flex: 1 }}>
        {icon && (
          <View
            style={{
              width: 34,
              height: 34,
              borderRadius: 10,
              backgroundColor: value > 0 ? color + "20" : "#f1f5f9",
              alignItems: "center",
              justifyContent: "center",
              marginRight: 10,
            }}
          >
            <Ionicons name={icon} size={17} color={value > 0 ? color : C.sub} />
          </View>
        )}
        <View style={{ flex: 1 }}>
          <Text style={{ color: C.text, fontWeight: "700", fontSize: 14 }}>{label}</Text>
          {sublabel && <Text style={{ color: C.sub, fontSize: 11, marginTop: 1 }}>{sublabel}</Text>}
        </View>
      </View>

      <View style={{ flexDirection: "row", alignItems: "center", gap: 2 }}>
        <TouchableOpacity
          onPress={() => onChange(Math.max(0, value - 1))}
          activeOpacity={0.7}
          style={{
            width: 32,
            height: 32,
            borderRadius: 9,
            backgroundColor: "#ffffff",
            alignItems: "center",
            justifyContent: "center",
            borderWidth: 1,
            borderColor: C.border,
          }}
        >
          <Ionicons name="remove" size={15} color={C.text} />
        </TouchableOpacity>

        <Text
          style={{
            color: value > 0 ? color : C.sub,
            fontWeight: "900",
            fontSize: 17,
            minWidth: 36,
            textAlign: "center",
          }}
        >
          {value}
        </Text>

        <TouchableOpacity
          onPress={() => onChange(Math.min(max, value + 1))}
          activeOpacity={0.7}
          style={{
            width: 32,
            height: 32,
            borderRadius: 9,
            backgroundColor: "#ffffff",
            alignItems: "center",
            justifyContent: "center",
            borderWidth: 1,
            borderColor: C.border,
          }}
        >
          <Ionicons name="add" size={15} color={C.text} />
        </TouchableOpacity>
      </View>
    </View>
  );
};

// ─── ProgressBar ──────────────────────────────────────────────
export const ProgressBar: React.FC<{
  progress: number;
  color?: string;
  height?: number;
  animated?: boolean;
  style?: StyleProp<ViewStyle>;
}> = ({ progress, color = C.primary, height = 8, animated = true, style }) => {
  const clamped = Math.max(0, Math.min(1, progress));
  const widthAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (animated) {
      Animated.timing(widthAnim, {
        toValue: clamped * 100,
        duration: 600,
        useNativeDriver: false,
      }).start();
    } else {
      widthAnim.setValue(clamped * 100);
    }
  }, [clamped]);

  return (
    <View
      style={[
        {
          height,
          backgroundColor: "#e2e8f0",
          borderRadius: height / 2,
          overflow: "hidden",
          width: "100%",
        },
        style,
      ]}
    >
      <Animated.View
        style={{
          height: "100%",
          width: widthAnim.interpolate({
            inputRange: [0, 100],
            outputRange: ["0%", "100%"],
          }),
          backgroundColor: color,
          borderRadius: height / 2,
        }}
      />
    </View>
  );
};

// ─── SkeletonBox ──────────────────────────────────────────────
export const SkeletonBox: React.FC<{
  width?: number | string;
  height?: number;
  borderRadius?: number;
  style?: StyleProp<ViewStyle>;
}> = ({ width = "100%", height = 20, borderRadius = 8, style }) => {
  const shimmer = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(shimmer, { toValue: 1, duration: 800, useNativeDriver: false }),
        Animated.timing(shimmer, { toValue: 0, duration: 800, useNativeDriver: false }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, []);

  const bgColor = shimmer.interpolate({
    inputRange: [0, 1],
    outputRange: ["#e2e8f0", "#f1f5f9"],
  });

  return (
    <Animated.View
      style={[
        {
          width: width as any,
          height,
          borderRadius,
          backgroundColor: bgColor,
        },
        style,
      ]}
    />
  );
};

// ─── Skeleton Loading Placeholder ─────────────────────────────
export const LoadingSkeleton: React.FC = () => (
  <View style={{ padding: 16 }}>
    <SkeletonBox height={28} width="60%" style={{ marginBottom: 6 }} />
    <SkeletonBox height={14} width="40%" style={{ marginBottom: 20 }} />
    <SkeletonBox height={140} borderRadius={20} style={{ marginBottom: 14 }} />
    <SkeletonBox height={14} width="35%" style={{ marginBottom: 8 }} />
    <SkeletonBox height={100} borderRadius={18} style={{ marginBottom: 14 }} />
    <SkeletonBox height={14} width="45%" style={{ marginBottom: 8 }} />
    <SkeletonBox height={120} borderRadius={18} />
  </View>
);

// ─── EmptyState ───────────────────────────────────────────────
export const EmptyState: React.FC<{
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  subtitle?: string;
  iconColor?: string;
  iconBg?: string;
  action?: React.ReactNode;
}> = ({ icon, title, subtitle, iconColor = C.sub, iconBg = "#f1f5f9", action }) => (
  <View style={{ alignItems: "center", justifyContent: "center", paddingVertical: 48, paddingHorizontal: 24 }}>
    <View
      style={{
        width: 80,
        height: 80,
        borderRadius: 24,
        backgroundColor: iconBg,
        alignItems: "center",
        justifyContent: "center",
        marginBottom: 16,
      }}
    >
      <Ionicons name={icon} size={40} color={iconColor} />
    </View>
    <Text style={{ color: C.text, fontSize: 18, fontWeight: "800", textAlign: "center" }}>{title}</Text>
    {subtitle && (
      <Text style={{ color: C.sub, fontSize: 14, textAlign: "center", marginTop: 6, maxWidth: 280, lineHeight: 20 }}>
        {subtitle}
      </Text>
    )}
    {action && <View style={{ marginTop: 18 }}>{action}</View>}
  </View>
);

// ─── ErrorBanner ──────────────────────────────────────────────
export const ErrorBanner: React.FC<{ msg: string | null; type?: "error" | "info" | "success" | "warning" }> = ({
  msg,
  type = "error",
}) => {
  if (!msg) return null;

  const map = {
    error:   { bg: C.redBg,    border: "#fecaca", icon: "alert-circle"        as keyof typeof Ionicons.glyphMap, col: C.red    },
    success: { bg: C.greenBg,  border: "#bbf7d0", icon: "checkmark-circle"    as keyof typeof Ionicons.glyphMap, col: C.green  },
    info:    { bg: C.primaryBg,border: "#bae6fd", icon: "information-circle"  as keyof typeof Ionicons.glyphMap, col: C.primary},
    warning: { bg: C.yellowBg, border: "#fde68a", icon: "warning"             as keyof typeof Ionicons.glyphMap, col: C.yellow },
  };
  const { bg, border, icon, col } = map[type];

  return (
    <View
      style={{
        backgroundColor: bg,
        padding: 14,
        borderRadius: 16,
        marginVertical: 8,
        borderWidth: 1,
        borderColor: border,
        flexDirection: "row",
        alignItems: "center",
      }}
    >
      <Ionicons name={icon} size={20} color={col} style={{ marginRight: 10 }} />
      <Text style={{ color: C.text, fontWeight: "600", fontSize: 13, flex: 1, lineHeight: 18 }}>{msg}</Text>
    </View>
  );
};

// ─── InfoRow ──────────────────────────────────────────────────
export const InfoRow: React.FC<{
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  value: string;
  iconColor?: string;
  onPress?: () => void;
}> = ({ icon, label, value, iconColor = C.primary, onPress }) => {
  const inner = (
    <View style={{ flexDirection: "row", alignItems: "center", paddingVertical: 10 }}>
      <View
        style={{
          width: 32,
          height: 32,
          borderRadius: 10,
          backgroundColor: C.primaryBg,
          alignItems: "center",
          justifyContent: "center",
          marginRight: 12,
        }}
      >
        <Ionicons name={icon} size={16} color={iconColor} />
      </View>
      <Text style={{ flex: 1, color: C.textSecondary, fontSize: 14, fontWeight: "600" }}>{label}</Text>
      <Text style={{ color: C.text, fontSize: 14, fontWeight: "700" }}>{value}</Text>
      {onPress && <Ionicons name="chevron-forward" size={16} color={C.sub} style={{ marginLeft: 6 }} />}
    </View>
  );

  if (onPress) return <TouchableOpacity onPress={onPress} activeOpacity={0.8}>{inner}</TouchableOpacity>;
  return inner;
};

// ─── AnimatedNumber ──────────────────────────────────────────
// Smooth count-up animation from 0 to a target number on mount/change.
export const AnimatedNumber: React.FC<{
  value: number;
  duration?: number;
  textStyle?: any;
  formatter?: (n: number) => string;
}> = ({ value, duration = 1100, textStyle, formatter }) => {
  const anim = useRef(new Animated.Value(0)).current;
  const [display, setDisplay] = useState(0);
  const idRef = useRef<string | null>(null);

  useEffect(() => {
    if (idRef.current) anim.removeListener(idRef.current);
    anim.setValue(0);
    idRef.current = anim.addListener(({ value: v }) => setDisplay(Math.round(v)));
    Animated.timing(anim, {
      toValue: value,
      duration,
      useNativeDriver: false,
    }).start();
    return () => { if (idRef.current) anim.removeListener(idRef.current); };
  }, [value]);

  return (
    <Text style={textStyle}>{formatter ? formatter(display) : display}</Text>
  );
};

// ─── MiniSparkBar ─────────────────────────────────────────────
// Tiny bar chart row for 7-day trends (used in outbreak cards).
export const MiniSparkBar: React.FC<{
  data: number[];
  color?: string;
  height?: number;
}> = ({ data, color = C.primary, height = 28 }) => {
  const max = Math.max(...data, 1);
  return (
    <View style={{ flexDirection: "row", alignItems: "flex-end", height, gap: 3 }}>
      {data.map((val, idx) => (
        <View
          key={idx}
          style={{
            flex: 1,
            height: Math.max(4, (val / max) * height),
            backgroundColor: idx === data.length - 1 ? color : `${color}60`,
            borderRadius: 3,
          }}
        />
      ))}
    </View>
  );
};

// ─── NurseCalendarStrip ───────────────────────────────────────
// Horizontal 7-day strip showing which days have nurse log entries.
type CalDay = {
  date: string;
  has_entry: boolean;
  fever: number;
  cough: number;
  cold: number;
  diarrhea: number;
  vomiting: number;
  others: number;
};

export const NurseCalendarStrip: React.FC<{
  history: CalDay[];
  onDayPress?: (day: CalDay) => void;
}> = ({ history, onDayPress }) => {
  const DAY_ABBR = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  return (
    <View style={{ flexDirection: "row", gap: 6, marginTop: 2 }}>
      {history.map((day) => {
        const d = new Date(day.date + "T00:00:00");
        const dayName = DAY_ABBR[d.getDay()];
        const dateNum = d.getDate();
        const total = day.fever + day.cough + day.cold + day.diarrhea + day.vomiting + day.others;
        const isToday = day.date === new Date().toISOString().slice(0, 10);

        const bg = isToday
          ? C.primaryBg
          : day.has_entry
          ? total > 20 ? C.redBg : total > 8 ? C.yellowBg : C.greenBg
          : "#f8fafc";

        const dotColor = isToday
          ? C.primary
          : day.has_entry
          ? total > 20 ? C.red : total > 8 ? C.yellow : C.green
          : C.border;

        const textColor = isToday ? C.primary : day.has_entry ? C.text : C.textMuted;

        return (
          <TouchableOpacity
            key={day.date}
            onPress={() => onDayPress?.(day)}
            activeOpacity={0.75}
            style={{ flex: 1 }}
          >
            <View
              style={{
                backgroundColor: bg,
                borderRadius: 14,
                padding: 8,
                alignItems: "center",
                borderWidth: isToday ? 2 : 1,
                borderColor: isToday ? C.primary : day.has_entry ? dotColor + "60" : C.border,
              }}
            >
              <Text style={{ color: C.sub, fontSize: 9, fontWeight: "700", marginBottom: 3 }}>
                {dayName}
              </Text>
              <Text style={{ color: textColor, fontSize: 13, fontWeight: "900" }}>{dateNum}</Text>
              <View
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: 4,
                  backgroundColor: dotColor,
                  marginTop: 5,
                }}
              />
              {day.has_entry && total > 0 && (
                <Text style={{ color: dotColor, fontSize: 9, fontWeight: "800", marginTop: 2 }}>
                  {total}
                </Text>
              )}
            </View>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};
