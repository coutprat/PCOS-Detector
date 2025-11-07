import { AlertCircle, X } from "lucide-react";
import { useEffect } from "react";

interface AlertProps {
  message: string;
  onClose?: () => void;
  variant?: "error" | "warning" | "info";
  autoClose?: boolean;
  autoCloseDelay?: number;
}

export default function Alert({
  message,
  onClose,
  variant = "info",
  autoClose = false,
  autoCloseDelay = 5000,
}: AlertProps) {
  useEffect(() => {
    if (autoClose && onClose) {
      const timer = setTimeout(onClose, autoCloseDelay);
      return () => clearTimeout(timer);
    }
  }, [autoClose, autoCloseDelay, onClose]);

  const variants = {
    error: "bg-red-500/20 border-red-500/50 text-red-300",
    warning: "bg-yellow-500/20 border-yellow-500/50 text-yellow-300",
    info: "bg-blue-500/20 border-blue-500/50 text-blue-300",
  };

  return (
    <div
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 rounded-lg border p-4 shadow-lg flex items-center gap-3 min-w-[300px] max-w-[90vw] ${variants[variant]}`}
      role="alert"
    >
      <AlertCircle className="w-5 h-5 flex-shrink-0" />
      <p className="flex-1">{message}</p>
      {onClose && (
        <button
          onClick={onClose}
          className="p-1 hover:bg-white/10 rounded transition-colors"
          aria-label="Close alert"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
