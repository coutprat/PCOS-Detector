import { useState } from "react";
import { HelpCircle } from "lucide-react";

interface ThresholdSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export default function ThresholdSlider({
  value,
  onChange,
  min = 0.05,
  max = 0.95,
  step = 0.01,
}: ThresholdSliderProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-medium flex items-center gap-2">
          Prediction Threshold
          <div className="relative">
            <button
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              className="text-white/60 hover:text-white transition-colors"
              aria-label="Threshold help"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
            {showTooltip && (
              <div className="absolute bottom-full left-0 mb-2 p-3 bg-slate-800 border border-white/20 rounded-lg shadow-lg w-64 z-10 text-xs">
                The threshold determines when a case is classified as Positive
                or Negative based on the prediction probability.
              </div>
            )}
          </div>
        </label>
        <span className="text-lg font-bold bg-white/10 px-4 py-1 rounded-full">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-blue-500"
        aria-label="Threshold slider"
      />
      <div className="flex justify-between text-xs text-white/60 mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
