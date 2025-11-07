import { AlertTriangle } from "lucide-react";
import { USE_MOCK } from "../lib/env";
import { useState } from "react";

export default function MockModeBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (!USE_MOCK || dismissed) return null;

  return (
    <div className="bg-yellow-500/20 border-b border-yellow-500/50 sticky top-16 z-30">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            <span className="text-sm text-yellow-300">
              Mock Mode: Backend not detected. Using simulated predictions.
            </span>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-yellow-300 hover:text-yellow-200 transition-colors text-sm font-semibold"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
