import axios from "axios";

// In dev, Vite proxies /api and /health to the backend (see vite.config.js).
const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  (import.meta.env.DEV ? "" : "http://localhost:8000");
const api = axios.create({
  baseURL: API_BASE,
  timeout: 90000,
  headers: { "Content-Type": "application/json" },
});

export async function askQuestion(payload) {
  const { data } = await api.post("/api/ask", payload);
  return data;
}

export async function askByQuestion(question) {
  const { data } = await api.post("/ask", { question });
  return data;
}

export async function planItinerary(payload) {
  const { data } = await api.post("/api/itinerary", payload);
  return data;
}

export async function fetchHotels(location) {
  const path = location ? `/hotels/${encodeURIComponent(location)}` : "/hotels";
  const { data } = await api.get(path);
  return data;
}

export async function checkHealth() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
