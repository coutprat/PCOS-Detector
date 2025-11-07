import { CheckCircle2, XCircle } from "lucide-react";

interface MetricBadgeProps {
  label: string;
  probability: number;
}

export default function MetricBadge({ label, probability }: MetricBadgeProps) {
  const isPositive = label === "Positive";

  return (
    <div
      className={`metric-badge ${
        isPositive ? "metric-badge-positive" : "metric-badge-negative"
      }`}
    >
      {isPositive ? (
        <CheckCircle2 className="w-4 h-4" />
      ) : (
        <XCircle className="w-4 h-4" />
      )}
      <span>{label}</span>
      <span className="ml-2 opacity-80">
        ({(probability * 100).toFixed(2)}%)
      </span>
    </div>
  );
}
