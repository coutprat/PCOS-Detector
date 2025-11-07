import { useState } from "react";
import Card from "../components/Card";
import UploadBox from "../components/UploadBox";
import Spinner from "../components/Spinner";
import Alert from "../components/Alert";
import { predictImage } from "../lib/api";

export default function ImageOnly() {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState<number>(0.44);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    probability: number;
    label: string;
    threshold: number;
    checkpoint_loaded?: boolean;
  } | null>(null);

  const handlePredict = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictImage(file, threshold);
      setResult(res);
    } catch (e: any) {
      setError(e?.message || "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-3xl">
      {error && (
        <Alert message={error} onClose={() => setError(null)} variant="error" />
      )}

      <h1 className="text-4xl font-bold mb-2">Image-Only Prediction</h1>
      <p className="text-white/70 mb-8">
        Upload an ultrasound image and set a threshold to get a prediction.
      </p>

      <Card>
        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-bold mb-4">Upload Image</h2>
            <UploadBox
              onFile={setFile}
              currentFile={file}
              accept="image/*"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Threshold</label>
            <input
              type="number"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="input-field max-w-xs"
            />
          </div>

          <div>
            <button
              onClick={handlePredict}
              disabled={!file || isLoading}
              className="btn-primary"
            >
              {isLoading ? <Spinner className="w-5 h-5 mx-auto" /> : "Predict"}
            </button>
          </div>
        </div>
      </Card>

      {result && (
        <Card className="mt-8">
          <h2 className="text-2xl font-bold mb-4">Results</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-sm text-white/60 mb-1">Probability</p>
              <p className="text-2xl font-bold">
                {result.probability.toFixed(3)}
              </p>
              <div className="mt-3 h-2 bg-white/10 rounded">
                <div
                  className="h-2 bg-blue-400/70 rounded transition-all"
                  style={{ width: `${Math.round(result.probability * 100)}%` }}
                />
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-sm text-white/60 mb-1">Label</p>
              <p className="text-2xl font-bold">{result.label}</p>
            </div>
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-sm text-white/60 mb-1">Threshold Used</p>
              <p className="text-2xl font-bold">{result.threshold}</p>
            </div>
          </div>
          <p className="text-xs text-white/60 mt-6 italic">
            Research prototype — not for clinical use. Model: ResNet-50 (SAM).
            Calibrated T*=0.988.
          </p>
        </Card>
      )}
    </div>
  );
}
