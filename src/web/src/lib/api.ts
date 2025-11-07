import { API_BASE } from "./env";
import type { EHRData } from "./ehrSchema";

export interface HealthResponse {
  status: string;
  device: "cuda" | "cpu";
  ckpt_exists: boolean;
}

export interface PredictResponse {
  probability: number;
  label: string;
  threshold: number;
  checkpoint_loaded?: boolean;
}

export interface HybridResponse {
  probability: number;
  label: string;
  threshold: number;
  image_prob: number;
  ehr_prob: number;
}

export interface CalibrateResponse {
  calibrated: number;
}

export interface GradCamResponse {
  overlay_url?: string;
}

async function getJson<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

function normalizePredict(json: any): PredictResponse {
  // Accept various backend shapes and normalize to PredictResponse
  const prob =
    json?.probability ?? json?.prob ?? json?.calibrated_prob ?? json?.p ?? 0;
  const thr = json?.threshold ?? json?.thr ?? 0.5;
  const predRaw = json?.label ?? json?.pred;
  let label: string;
  if (typeof predRaw === "string") label = predRaw;
  else if (typeof predRaw === "boolean") label = predRaw ? "PCOS Positive" : "PCOS Negative";
  else label = (Number(prob) >= Number(thr)) ? "PCOS Positive" : "PCOS Negative";
  return {
    probability: Number(prob),
    threshold: Number(thr),
    label,
    checkpoint_loaded: json?.checkpoint_loaded,
  };
}

export const getHealth = async (): Promise<HealthResponse> => {
  return getJson<HealthResponse>(`${API_BASE}/health`, { method: "GET" });
};

export const reloadModel = async (): Promise<{ reloaded: boolean }> => {
  return getJson<{ reloaded: boolean }>(`${API_BASE}/reload`, {
    method: "POST",
  });
};

export const predictImage = async (
  file: File,
  threshold?: number
): Promise<PredictResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  const url =
    typeof threshold === "number"
      ? `${API_BASE}/predict?threshold=${encodeURIComponent(threshold)}`
      : `${API_BASE}/predict`;
  return getJson<PredictResponse>(url, {
    method: "POST",
    body: formData,
  });
};

export const predictEhr = async (
  ehr: EHRData,
  threshold?: number
): Promise<PredictResponse> => {
  // Try multiple endpoint/shape combinations to be robust
  const tries: Array<{ url: string; body: any }> = [];
  const flat = { ...ehr, ...(typeof threshold === "number" ? { threshold } : {}) };
  const wrapped = { ehr, ...(typeof threshold === "number" ? { threshold } : {}) };
  tries.push({ url: `${API_BASE}/predict/ehr`, body: flat });
  tries.push({ url: `${API_BASE}/predict/ehr`, body: wrapped });
  tries.push({ url: `${API_BASE}/predict_ehr`, body: flat });
  tries.push({ url: `${API_BASE}/predict_ehr`, body: wrapped });

  let lastErr: unknown;
  for (const t of tries) {
    try {
      const raw = await getJson<any>(t.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(t.body),
      });
      return normalizePredict(raw);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
};

export const predictHybrid = async (
  file: File,
  ehr: EHRData,
  threshold?: number
): Promise<HybridResponse> => {
  // multipart: file + ehr_json; optional threshold
  const formData = new FormData();
  formData.append("file", file);
  formData.append("ehr_json", JSON.stringify(ehr));
  if (typeof threshold === "number") {
    formData.append("threshold", String(threshold));
  }
  return getJson<HybridResponse>(`${API_BASE}/predict_hybrid`, {
    method: "POST",
    body: formData,
  });
};

// Aliases to match requested names
export const predictEHR = predictEhr;

export const calibrate = async (
  probability: number
): Promise<CalibrateResponse> => {
  return getJson<CalibrateResponse>(`${API_BASE}/calibrate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ probability }),
  });
};

export const gradcam = async (
  file: File
): Promise<{ overlayUrl?: string; blobUrl?: string }> => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/gradcam`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const json = (await res.json()) as GradCamResponse;
    return { overlayUrl: json.overlay_url };
  }
  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  return { blobUrl };
};
