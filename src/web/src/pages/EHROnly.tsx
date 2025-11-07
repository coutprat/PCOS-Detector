import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import Card from "../components/Card";
import MetricBadge from "../components/MetricBadge";
import ThresholdSlider from "../components/ThresholdSlider";
import Spinner from "../components/Spinner";
import Alert from "../components/Alert";
import { predictEhr } from "../lib/api";
import { ehrSchema, type EHRData, defaultEhr } from "../lib/ehrSchema";
import { formatProbability } from "../lib/utils";

export default function EHROnly() {
  const [threshold, setThreshold] = useState(0.5);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{
    probability: number;
    label: string;
    threshold: number;
  } | null>(null);
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
    setIsLoading(true);
    setError(null);
    try {
      const response = await predictEhr(data, threshold);
      setResult(response);
    } catch (err) {
      setError("Failed to generate prediction. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      {error && (
        <Alert message={error} onClose={() => setError(null)} variant="error" />
      )}

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-4xl font-bold mb-2">EHR-Only Prediction</h1>
        <p className="text-white/70 mb-8">
          Fill out medical history and lab values for risk assessment
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <form onSubmit={handleSubmit(onSubmit)}>
            {/* Demographics */}
            <Card>
              <h2 className="text-2xl font-bold mb-6 pb-4 border-b border-white/10">
                Demographics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Age (years)
                  </label>
                  <input
                    type="number"
                    step="1"
                    {...register("demographics.age", { valueAsNumber: true })}
                    className="input-field"
                  />
                  {errors.demographics?.age && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.demographics.age.message}
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    BMI (kg/m²)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    {...register("demographics.bmi", { valueAsNumber: true })}
                    className="input-field"
                  />
                  {errors.demographics?.bmi && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.demographics.bmi.message}
                    </p>
                  )}
                </div>
              </div>
            </Card>

            {/* Endocrine */}
            <Card>
              <h2 className="text-2xl font-bold mb-6 pb-4 border-b border-white/10">
                Endocrine Markers
              </h2>
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
                  {errors.endocrine?.amh && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.endocrine.amh.message}
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Total Testosterone (ng/dL)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    {...register("endocrine.total_testosterone", {
                      valueAsNumber: true,
                    })}
                    className="input-field"
                  />
                  {errors.endocrine?.total_testosterone && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.endocrine.total_testosterone.message}
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Hirsutism (Ferriman-Gallwey, 0-36)
                  </label>
                  <input
                    type="number"
                    step="1"
                    {...register("endocrine.hirsutism", {
                      valueAsNumber: true,
                    })}
                    className="input-field"
                  />
                  {errors.endocrine?.hirsutism && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.endocrine.hirsutism.message}
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Acne Severity (0-4)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    {...register("endocrine.acne_severity", {
                      valueAsNumber: true,
                    })}
                    className="input-field"
                  />
                  {errors.endocrine?.acne_severity && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.endocrine.acne_severity.message}
                    </p>
                  )}
                </div>
              </div>
            </Card>

            {/* Menstrual */}
            <Card>
              <h2 className="text-2xl font-bold mb-6 pb-4 border-b border-white/10">
                Menstrual History
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Cycle Regularity
                  </label>
                  <select
                    {...register("menstrual.cycle_regularity")}
                    className="input-field"
                  >
                    <option value="Regular">Regular</option>
                    <option value="Irregular">Irregular</option>
                  </select>
                  {errors.menstrual?.cycle_regularity && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.menstrual.cycle_regularity.message}
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Avg Cycle Length (days)
                  </label>
                  <input
                    type="number"
                    step="1"
                    {...register("menstrual.avg_cycle_length_days", {
                      valueAsNumber: true,
                    })}
                    className="input-field"
                  />
                  {errors.menstrual?.avg_cycle_length_days && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.menstrual.avg_cycle_length_days.message}
                    </p>
                  )}
                </div>
              </div>
            </Card>

            {/* Metabolic */}
            <Card>
              <h2 className="text-2xl font-bold mb-6 pb-4 border-b border-white/10">
                Metabolic Markers
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Fasting Insulin (uIU/mL)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    {...register("metabolic.fasting_insulin", {
                      valueAsNumber: true,
                    })}
                    className="input-field"
                  />
                  {errors.metabolic?.fasting_insulin && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.metabolic.fasting_insulin.message}
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Fasting Glucose (mg/dL)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    {...register("metabolic.fasting_glucose", {
                      valueAsNumber: true,
                    })}
                    className="input-field"
                  />
                  {errors.metabolic?.fasting_glucose && (
                    <p className="mt-1 text-xs text-red-400">
                      {errors.metabolic.fasting_glucose.message}
                    </p>
                  )}
                </div>
              </div>
            </Card>

            <Card>
              <ThresholdSlider value={threshold} onChange={setThreshold} />
            </Card>

            <div className="sticky bottom-0 bg-gradient-medical py-4 -mx-4 px-4 border-t border-white/10">
              <button
                type="submit"
                disabled={isLoading}
                className="btn-primary w-full"
              >
                {isLoading ? (
                  <Spinner className="w-5 h-5 mx-auto" />
                ) : (
                  "Generate Prediction"
                )}
              </button>
            </div>
          </form>
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
                    <p className="text-sm text-white/60 mb-1">Probability</p>
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
