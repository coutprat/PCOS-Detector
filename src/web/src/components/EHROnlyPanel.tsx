import React, { useMemo, useState } from "react";

type Cycle = "regular" | "irregular";
type Hirsutism = "none" | "mild" | "moderate" | "severe";

type EHRFeatures = {
  age: number;
  bmi: number;
  cycleRegularity: Cycle;
  hirsutism: Hirsutism;
  lh: number;
  fsh: number;
  testosterone: number;
  fastingInsulin: number;
  fastingGlucose: number;
  amh: number;
};

type Contributor = { name: keyof EHRFeatures; value: number };
type EHRResult = {
  probability: number;       // raw 0..1
  calibrated: number;        // demo-calibrated
  threshold: number;         // default 0.35
  label: "PCOS Positive" | "PCOS Negative";
  topContributors: Contributor[]; // signed contributions
  summary: string;
};

// --- Demo defaults (realistic ranges, purely illustrative) ---
const DEMO_DEFAULTS: EHRFeatures = {
  age: 26,
  bmi: 28.4,
  cycleRegularity: "irregular",
  hirsutism: "moderate",
  lh: 12.5,
  fsh: 5.1,
  testosterone: 62,
  fastingInsulin: 18,
  fastingGlucose: 96,
  amh: 6.2,
};

// Simple feature meta for validation & tooltips
const META: Record<keyof EHRFeatures, { min?: number; max?: number; unit?: string; help?: string }> = {
  age: { min: 15, max: 55, unit: "years" },
  bmi: { min: 12, max: 60, unit: "kg/m²" },
  cycleRegularity: { help: "Usual cycle pattern." },
  hirsutism: { help: "Clinician/subjective assessment." },
  lh: { min: 0, max: 40, unit: "IU/L" },
  fsh: { min: 0, max: 40, unit: "IU/L" },
  testosterone: { min: 0, max: 300, unit: "ng/dL" },
  fastingInsulin: { min: 0, max: 300, unit: "µIU/mL" },
  fastingGlucose: { min: 40, max: 300, unit: "mg/dL" },
  amh: { min: 0, max: 20, unit: "ng/mL" },
};

// Fixed z-score means/stds (demo only, not medical advice)
const ZS: Record<keyof EHRFeatures, { mean: number; sd: number }> = {
  age: { mean: 29, sd: 8 },
  bmi: { mean: 25, sd: 6 },
  cycleRegularity: { mean: 0, sd: 1 },
  hirsutism: { mean: 0, sd: 1 },
  lh: { mean: 7.5, sd: 4 },
  fsh: { mean: 6.5, sd: 2.5 },
  testosterone: { mean: 35, sd: 18 },
  fastingInsulin: { mean: 9, sd: 6 },
  fastingGlucose: { mean: 92, sd: 10 },
  amh: { mean: 3.0, sd: 2.0 },
};

// Fixed weights for “mock SHAP-like” contribution (demo only)
const W: Record<keyof EHRFeatures, number> = {
  age: -0.12,
  bmi: 0.22,
  cycleRegularity: 0.35,   // irregular = +1, regular = 0
  hirsutism: 0.28,         // none=0, mild=0.4, moderate=0.7, severe=1
  lh: 0.18,
  fsh: -0.20,
  testosterone: 0.26,
  fastingInsulin: 0.24,
  fastingGlucose: 0.05,
  amh: 0.21,
};

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}
function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}
function encodeCycle(c: Cycle) {
  return c === "irregular" ? 1 : 0;
}
function encodeHirsutism(h: Hirsutism) {
  switch (h) {
    case "none": return 0;
    case "mild": return 0.4;
    case "moderate": return 0.7;
    case "severe": return 1;
  }
}

// Fake temperature scaling (demo-calibration)
function calibrate(prob: number, temperature = 1.2) {
  const logit = Math.log(prob / (1 - prob));
  const cooled = logit / temperature;
  return clamp01(sigmoid(cooled));
}

// Compute signed contributions (z-score * weight)
function computeContribs(f: EHRFeatures): Contributor[] {
  const z = {
    age: (f.age - ZS.age.mean) / ZS.age.sd,
    bmi: (f.bmi - ZS.bmi.mean) / ZS.bmi.sd,
    cycleRegularity: (encodeCycle(f.cycleRegularity) - ZS.cycleRegularity.mean) / ZS.cycleRegularity.sd,
    hirsutism: (encodeHirsutism(f.hirsutism) - ZS.hirsutism.mean) / ZS.hirsutism.sd,
    lh: (f.lh - ZS.lh.mean) / ZS.lh.sd,
    fsh: (f.fsh - ZS.fsh.mean) / ZS.fsh.sd,
    testosterone: (f.testosterone - ZS.testosterone.mean) / ZS.testosterone.sd,
    fastingInsulin: (f.fastingInsulin - ZS.fastingInsulin.mean) / ZS.fastingInsulin.sd,
    fastingGlucose: (f.fastingGlucose - ZS.fastingGlucose.mean) / ZS.fastingGlucose.sd,
    amh: (f.amh - ZS.amh.mean) / ZS.amh.sd,
  } as Record<keyof EHRFeatures, number>;

  const raw: Contributor[] = (Object.keys(W) as (keyof EHRFeatures)[])
    .map((k) => ({ name: k, value: z[k] * W[k] }));

  // Sort by absolute contribution, desc
  raw.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return raw;
}

// Map contributions to a probability for demo purposes
function scoreToProbability(contribs: Contributor[]): number {
  const bias = -0.1; // bias term (demo)
  const total = contribs.reduce((s, c) => s + c.value, bias);
  return clamp01(sigmoid(total));
}

function makeSummary(top: Contributor[], threshold: number, prob: number) {
  const dir = (v: number) => (v >= 0 ? "↑" : "↓");
  const drivers = top.slice(0, 3).map((c) => `${c.name}${dir(c.value)}`).join(", ");
  const stance = prob >= threshold ? "above" : "below";
  return `Risk drivers: ${drivers}. Estimated risk is ${stance} threshold (${threshold}).`;
}

const Field: React.FC<{
  label: string;
  unit?: string;
  value: string | number;
  onChange: (v: string) => void;
  type?: "number" | "text";
  min?: number;
  max?: number;
  disabled?: boolean;
}> = ({ label, unit, value, onChange, type = "number", min, max, disabled }) => (
  <label className="block">
    <span className="text-sm text-gray-700">{label}{unit ? ` (${unit})` : ""}</span>
    <input
      className="mt-1 w-full rounded-xl border px-3 py-2 shadow-sm focus:outline-none focus:ring"
      type={type}
      value={value}
      min={min}
      max={max}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
    />
  </label>
);

export default function EHROnlyPanel() {
  const [f, setF] = useState<EHRFeatures>(DEMO_DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<EHRResult | null>(null);
  const [threshold, setThreshold] = useState(0.35);

  const canRun = useMemo(() => !busy, [busy]);

  const run = async () => {
    setBusy(true);
    setResult(null);
    setProgress(0);

    // fake progress
    const start = Date.now();
    const totalMs = 2500;
    await new Promise<void>((resolve) => {
      const tick = () => {
        const t = Date.now() - start;
        const pct = Math.min(100, Math.round((t / totalMs) * 100));
        setProgress(pct);
        if (pct >= 100) resolve();
        else setTimeout(tick, 60);
      };
      tick();
    });

    // compute mock output
    const contribs = computeContribs(f);
    const prob = scoreToProbability(contribs);
    const calibrated = calibrate(prob, 1.2);
    const label = calibrated >= threshold ? "PCOS Positive" : "PCOS Negative";
    const topFive = contribs.slice(0, 5);
    const summary = makeSummary(topFive, threshold, calibrated);

    setResult({ probability: prob, calibrated, threshold, label, topContributors: topFive, summary });
    setBusy(false);
  };

  const reset = () => {
    setF(DEMO_DEFAULTS);
    setResult(null);
    setProgress(0);
    setBusy(false);
    setThreshold(0.35);
  };

  // Helpers for updating numeric vs categorical
  const num = (k: keyof EHRFeatures) => ({
    value: String(f[k] as number),
    onChange: (v: string) => {
      const parsed = Number(v);
      if (Number.isFinite(parsed)) setF({ ...f, [k]: parsed });
      else setF({ ...f, [k]: 0 });
    },
    min: META[k].min,
    max: META[k].max,
    unit: META[k].unit,
  });

  return (
    <div className="mx-auto max-w-3xl p-4">
      <h1 className="text-2xl font-semibold mb-2">EHR-Only Screening (Demo)</h1>
      <p className="text-gray-600 mb-4">
        Enter EHR variables or use the prefilled demo values. Click <span className="font-medium">Run EHR-only</span> to simulate inference and see calibrated risk and top contributors.
      </p>

      {/* Processing banner */}
      {busy && (
        <div className="mb-4 rounded-xl bg-blue-50 p-3 border border-blue-200">
          <div className="mb-1 text-sm font-medium text-blue-900">Processing EHR data…</div>
          <div className="h-2 w-full rounded bg-blue-100">
            <div className="h-2 rounded bg-blue-500" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {/* Form */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <Field label="Age" {...num("age")} disabled={busy} />
        <Field label="BMI" {...num("bmi")} disabled={busy} />

        <label className="block">
          <span className="text-sm text-gray-700">Cycle Regularity</span>
          <select
            className="mt-1 w-full rounded-xl border px-3 py-2 shadow-sm focus:outline-none focus:ring"
            value={f.cycleRegularity}
            onChange={(e) => setF({ ...f, cycleRegularity: e.target.value as Cycle })}
            disabled={busy}
          >
            <option value="regular">regular</option>
            <option value="irregular">irregular</option>
          </select>
        </label>

        <label className="block">
          <span className="text-sm text-gray-700">Hirsutism</span>
          <select
            className="mt-1 w-full rounded-xl border px-3 py-2 shadow-sm focus:outline-none focus:ring"
            value={f.hirsutism}
            onChange={(e) => setF({ ...f, hirsutism: e.target.value as Hirsutism })}
            disabled={busy}
          >
            <option value="none">none</option>
            <option value="mild">mild</option>
            <option value="moderate">moderate</option>
            <option value="severe">severe</option>
          </select>
        </label>

        <Field label="LH" {...num("lh")} disabled={busy} />
        <Field label="FSH" {...num("fsh")} disabled={busy} />
        <Field label="Testosterone" {...num("testosterone")} disabled={busy} />
        <Field label="Fasting Insulin" {...num("fastingInsulin")} disabled={busy} />
        <Field label="Fasting Glucose" {...num("fastingGlucose")} disabled={busy} />
        <Field label="AMH" {...num("amh")} disabled={busy} />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <button
          className="rounded-2xl bg-blue-600 px-4 py-2 text-white shadow hover:bg-blue-700 disabled:opacity-50"
          onClick={run}
          disabled={!canRun}
        >
          Run EHR-only
        </button>
        <button
          className="rounded-2xl bg-gray-100 px-4 py-2 shadow hover:bg-gray-200"
          onClick={reset}
          disabled={busy}
        >
          Reset
        </button>

        <div className="ml-auto flex items-center gap-2">
          <span className="text-sm text-gray-700">Threshold</span>
          <input
            className="w-24 rounded-xl border px-3 py-2 shadow-sm focus:outline-none focus:ring"
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            disabled={busy}
          />
        </div>
      </div>

      {/* Result */}
      {result && (
        <div className="rounded-2xl border p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-lg font-semibold">Prediction</div>
              <div className="text-sm text-gray-600">EHR-only (demo)</div>
            </div>
            <div className={`rounded-xl px-3 py-1 text-sm ${result.label === "PCOS Positive" ? "bg-red-50 text-red-700 border border-red-200" : "bg-green-50 text-green-700 border border-green-200"}`}>
              {result.label}
            </div>
          </div>

          <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="rounded-xl bg-gray-50 p-3">
              <div className="text-xs text-gray-500">Probability</div>
              <div className="text-xl font-semibold">{result.probability.toFixed(3)}</div>
            </div>
            <div className="rounded-xl bg-gray-50 p-3">
              <div className="text-xs text-gray-500">Calibrated</div>
              <div className="text-xl font-semibold">{result.calibrated.toFixed(3)}</div>
            </div>
            <div className="rounded-xl bg-gray-50 p-3">
              <div className="text-xs text-gray-500">Threshold</div>
              <div className="text-xl font-semibold">{result.threshold.toFixed(2)}</div>
            </div>
          </div>

          <div className="mt-4">
            <div className="text-sm font-medium mb-2">Top Contributors (demo)</div>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {result.topContributors.map((c, idx) => (
                <li key={idx} className="rounded-lg border px-3 py-2 text-sm flex items-center justify-between">
                  <span className="capitalize">{String(c.name)}</span>
                  <span className={c.value >= 0 ? "text-red-700" : "text-green-700"}>
                    {c.value >= 0 ? "▲" : "▼"} {Math.abs(c.value).toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <p className="mt-3 text-sm text-gray-700">{result.summary}</p>
          <p className="mt-2 text-xs text-gray-500">
            Demo only — not medical advice. Calibrated value uses a fixed temperature; contributors are pseudo-SHAP based on a linear surrogate.
          </p>
        </div>
      )}
    </div>
  );
}



