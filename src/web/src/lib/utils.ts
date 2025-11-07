import { type ClassValue, clsx } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatProbability(prob: number, decimals: number = 4): string {
  return (prob * 100).toFixed(decimals);
}

export async function createBlobUrl(response: Response): Promise<string> {
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}
