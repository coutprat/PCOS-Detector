import { Loader2 } from "lucide-react";

export default function Spinner({
  className = "w-6 h-6",
}: {
  className?: string;
}) {
  return <Loader2 className={`animate-spin ${className}`} />;
}
