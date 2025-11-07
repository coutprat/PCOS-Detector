import type { EHRData } from "./ehrSchema";

// Deterministic hashing function for stable mock probabilities
function hash(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    const char = input.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

function hashToProbability(input: string): number {
  const h = hash(input);
  // Normalize to 0-1 range with some bias towards 0.3-0.7 for realism
  const normalized = (h % 10000) / 10000;
  return 0.2 + normalized * 0.6; // Range: 0.2 to 0.8
}

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

export const mockHealth = (): HealthResponse => ({
  status: "ok",
  device: "cpu",
  ckpt_exists: true,
});

export const mockPredictImage = async (
  file: File,
  threshold: number
): Promise<PredictResponse> => {
  const input = `${file.name}-${file.size}-${threshold}`;
  const probability = hashToProbability(input);
  return {
    probability,
    label: probability >= threshold ? "Positive" : "Negative",
    threshold,
    checkpoint_loaded: true,
  };
};

export const mockPredictEhr = async (
  ehr: EHRData,
  threshold: number
): Promise<PredictResponse> => {
  const input = JSON.stringify(ehr) + threshold;
  const probability = hashToProbability(input);
  return {
    probability,
    label: probability >= threshold ? "Positive" : "Negative",
    threshold,
  };
};

export const mockPredictHybrid = async (
  file: File,
  ehr: EHRData,
  threshold: number
): Promise<HybridResponse> => {
  const imageInput = `${file.name}-${file.size}`;
  const ehrInput = JSON.stringify(ehr);
  const imageProb = hashToProbability(imageInput);
  const ehrProb = hashToProbability(ehrInput);

  // Weighted fusion: 60% image, 40% EHR
  const fusedProb = imageProb * 0.6 + ehrProb * 0.4;

  return {
    probability: fusedProb,
    label: fusedProb >= threshold ? "Positive" : "Negative",
    threshold,
    image_prob: imageProb,
    ehr_prob: ehrProb,
  };
};

export const mockCalibrate = async (
  probability: number
): Promise<CalibrateResponse> => {
  // Temperature scaling: identity for simplicity
  return {
    calibrated: probability,
  };
};

export const mockGradCam = async (file: File): Promise<{ blobUrl: string }> => {
  // Generate a mock heatmap-like blob
  const canvas = document.createElement("canvas");
  canvas.width = 400;
  canvas.height = 400;
  const ctx = canvas.getContext("2d")!;

  // Create a gradient overlay
  const gradient = ctx.createRadialGradient(200, 200, 0, 200, 200, 200);
  gradient.addColorStop(0, "rgba(255, 0, 0, 0.8)");
  gradient.addColorStop(0.5, "rgba(255, 255, 0, 0.4)");
  gradient.addColorStop(1, "rgba(0, 0, 255, 0.1)");

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 400, 400);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      const blobUrl = URL.createObjectURL(blob!);
      resolve({ blobUrl });
    });
  });
};
