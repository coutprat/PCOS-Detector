export const API_BASE =
  (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";
export const USE_MOCK = ((import.meta as any).env?.VITE_USE_MOCK ?? "0") === "1";
