import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import Card from "../components/Card";
import UploadBox from "../components/UploadBox";
import ThresholdSlider from "../components/ThresholdSlider";
import MetricBadge from "../components/MetricBadge";
import GradCamViewer from "../components/GradCamViewer";
import Spinner from "../components/Spinner";
import Alert from "../components/Alert";
import { predictHybrid, gradcam } from "../lib/api";
import { ehrSchema, type EHRData, defaultEhr } from "../lib/ehrSchema";
import { formatProbability } from "../lib/utils";

export default function Hybrid() {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{
    probability: number;
    label: string;
    threshold: number;
    image_prob: number;
    ehr_prob: number;
  } | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<EHRData>({
    resolver: zodResolver(ehrSchema),
    defaultValues: defaultEhr,
  });

  const onSubmit = async (data: EHRData) => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await predictHybrid(file, data, threshold);
      setResult(response);
      setOverlayUrl(null);
      setBlobUrl(null);
    } catch (err) {
      setError("Failed to generate prediction. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateGradCam = async () => {
    if (!file) return;

    try {
      const response = await gradcam(file);
      if (response.overlayUrl) {
        setOverlayUrl(response.overlayUrl);
      } else if (response.blobUrl) {
        setBlobUrl(response.blobUrl);
      }
    } catch (err) {
      setError("Failed to generate Grad-CAM overlay. Please try again.");
    }
  };

  const baseImageUrl = file ? URL.createObjectURL(file) : "";

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {error && (
        <Alert message={error} onClose={() => setError(null)} variant="error" />
      )}

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-4xl font-bold mb-2">Hybrid Prediction</h1>
        <p className="text-white/70 mb-8">
          Combine image and EHR data for comprehensive AI-driven diagnosis
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <h2 className="text-2xl font-bold mb-6">Upload Image</h2>
            <UploadBox
              onFile={setFile}
              currentFile={file}
              accept="image/png,image/jpeg,image/jpg"
            />
          </Card>

          <form onSubmit={handleSubmit(onSubmit)}>
            <Card>
              <h2 className="text-2xl font-bold mb-6 pb-4 border-b border-white/10">
                Clinical Data
              </h2>
              <div className="space-y-6">
                {/* Demographics */}
                <div>
                  <h3 className="font-semibold mb-3 text-white/90">
                    Demographics
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Age
                      </label>
                      <input
                        type="number"
                        step="1"
                        {...register("demographics.age", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        BMI
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        {...register("demographics.bmi", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                  </div>
                </div>

                {/* Endocrine */}
                <div>
                  <h3 className="font-semibold mb-3 text-white/90">
                    Endocrine
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        AMH (ng/mL)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        {...register("endocrine.amh", { valueAsNumber: true })}
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Testosterone (ng/dL)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        {...register("endocrine.total_testosterone", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Hirsutism (0-36)
                      </label>
                      <input
                        type="number"
                        step="1"
                        {...register("endocrine.hirsutism", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Acne (0-4)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        {...register("endocrine.acne_severity", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                  </div>
                </div>

                {/* Menstrual */}
                <div>
                  <h3 className="font-semibold mb-3 text-white/90">
                    Menstrual
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Regularity
                      </label>
                      <select
                        {...register("menstrual.cycle_regularity")}
                        className="input-field"
                      >
                        <option value="Regular">Regular</option>
                        <option value="Irregular">Irregular</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Cycle Length (days)
                      </label>
                      <input
                        type="number"
                        step="1"
                        {...register("menstrual.avg_cycle_length_days", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                  </div>
                </div>

                {/* Metabolic */}
                <div>
                  <h3 className="font-semibold mb-3 text-white/90">
                    Metabolic
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Insulin (uIU/mL)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        {...register("metabolic.fasting_insulin", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Glucose (mg/dL)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        {...register("metabolic.fasting_glucose", {
                          valueAsNumber: true,
                        })}
                        className="input-field"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            <Card>
              <ThresholdSlider value={threshold} onChange={setThreshold} />
            </Card>

            <div className="sticky bottom-0 bg-gradient-medical py-4 -mx-4 px-4 border-t border-white/10">
              <button
                type="submit"
                disabled={!file || isLoading}
                className="btn-primary w-full"
              >
                {isLoading ? (
                  <Spinner className="w-5 h-5 mx-auto" />
                ) : (
                  "Generate Hybrid Prediction"
                )}
              </button>
            </div>
          </form>

          {file && (
            <Card>
              <GradCamViewer
                baseImageUrl={baseImageUrl}
                overlayUrl={overlayUrl || undefined}
                blobUrl={blobUrl || undefined}
                onGenerate={handleGenerateGradCam}
              />
            </Card>
          )}
        </div>

        <div>
          {result && (
            <Card className="sticky top-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold mb-4">Results</h2>
                <div className="space-y-4">
                  <MetricBadge
                    label={result.label}
                    probability={result.probability}
                  />
                  <div className="bg-white/5 rounded-lg p-4">
                    <p className="text-sm text-white/60 mb-1">
                      Fused Probability
                    </p>
                    <p className="text-3xl font-bold">
                      {formatProbability(result.probability, 4)}%
                    </p>
                    <div className="mt-3 h-2 bg-white/10 rounded">
                      <div
                        className="h-2 bg-blue-400/70 rounded transition-all"
                        style={{ width: `${Math.round(result.probability * 100)}%` }}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white/5 rounded-lg p-3">
                      <p className="text-xs text-white/60 mb-1">Image</p>
                      <p className="text-lg font-bold">
                        {formatProbability(result.image_prob, 2)}%
                      </p>
                      <div className="mt-2 h-2 bg-white/10 rounded">
                        <div
                          className="h-2 bg-blue-300/70 rounded transition-all"
                          style={{ width: `${Math.round(result.image_prob * 100)}%` }}
                        />
                      </div>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3">
                      <p className="text-xs text-white/60 mb-1">EHR</p>
                      <p className="text-lg font-bold">
                        {formatProbability(result.ehr_prob, 2)}%
                      </p>
                      <div className="mt-2 h-2 bg-white/10 rounded">
                        <div
                          className="h-2 bg-teal-300/70 rounded transition-all"
                          style={{ width: `${Math.round(result.ehr_prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-white/60 mt-6 italic">
                  Research prototype. Not for clinical use.
                </p>
              </motion.div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
