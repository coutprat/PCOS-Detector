import { useState, useEffect } from "react";
import { Eye, EyeOff } from "lucide-react";
import { motion } from "framer-motion";

interface GradCamViewerProps {
  baseImageUrl: string;
  overlayUrl?: string;
  blobUrl?: string;
  onGenerate: () => Promise<void>;
}

export default function GradCamViewer({
  baseImageUrl,
  overlayUrl,
  blobUrl,
  onGenerate,
}: GradCamViewerProps) {
  const [opacity, setOpacity] = useState(0.5);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    setIsGenerating(true);
    await onGenerate();
    setShowOverlay(true);
    setIsGenerating(false);
  };

  useEffect(() => {
    if (overlayUrl || blobUrl) {
      setShowOverlay(true);
    }
  }, [overlayUrl, blobUrl]);

  return (
    <div className="w-full space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Grad-CAM Visualization</h3>
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="btn-secondary"
        >
          {isGenerating ? "Generating..." : "Generate Grad-CAM"}
        </button>
      </div>

      {(overlayUrl || blobUrl) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative"
        >
          <div className="relative w-full aspect-square rounded-lg overflow-hidden border border-white/20">
            <img
              src={baseImageUrl}
              alt="Base ultrasound"
              className="w-full h-full object-contain"
            />
            {showOverlay && (overlayUrl || blobUrl) && (
              <img
                src={overlayUrl || blobUrl}
                alt="Grad-CAM overlay"
                className="absolute inset-0 w-full h-full object-contain"
                style={{ opacity }}
              />
            )}
          </div>

          <div className="mt-4 flex items-center gap-4">
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
            >
              {showOverlay ? (
                <Eye className="w-4 h-4" />
              ) : (
                <EyeOff className="w-4 h-4" />
              )}
              <span>Toggle Overlay</span>
            </button>
            <div className="flex-1 flex items-center gap-4">
              <label className="text-sm whitespace-nowrap">Opacity</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="flex-1 accent-blue-500"
              />
              <span className="text-sm w-12">{Math.round(opacity * 100)}%</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
