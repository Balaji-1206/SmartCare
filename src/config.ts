import { Platform } from "react-native";

export const API_BASE =
  Platform.select({
    android: "http://10.0.2.2:8000",
    ios: "http://127.0.0.1:8000",
    default: "http://127.0.0.1:8000",
  })!;
