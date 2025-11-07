import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Stethoscope } from "lucide-react";
import { getHealth } from "../lib/api";
import type { HealthResponse } from "../lib/api";

export default function Navbar() {
  const location = useLocation();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await getHealth();
        setHealth(data);
        setOffline(false);
      } catch (error) {
        setOffline(true);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHealth();
  }, []);

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-white/5 backdrop-blur-md border-b border-white/10 sticky top-0 z-40">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-teal-500 rounded-lg group-hover:scale-110 transition-transform">
              <Stethoscope className="w-6 h-6" />
            </div>
            <span className="text-2xl font-bold">PCOS Lab</span>
          </Link>

          <div className="flex items-center gap-8">
            <div className="hidden md:flex gap-1">
              <Link
                to="/image"
                className={`px-4 py-2 rounded-lg transition-colors ${
                  isActive("/image")
                    ? "bg-blue-500/20 text-blue-300"
                    : "hover:bg-white/10"
                }`}
              >
                Image
              </Link>
              <Link
                to="/ehr"
                className={`px-4 py-2 rounded-lg transition-colors ${
                  isActive("/ehr")
                    ? "bg-blue-500/20 text-blue-300"
                    : "hover:bg-white/10"
                }`}
              >
                EHR
              </Link>
              <Link
                to="/hybrid"
                className={`px-4 py-2 rounded-lg transition-colors ${
                  isActive("/hybrid")
                    ? "bg-blue-500/20 text-blue-300"
                    : "hover:bg-white/10"
                }`}
              >
                Hybrid
              </Link>
            </div>

            {!isLoading && (
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-white/10 rounded-full text-xs">
                {offline || !health ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-red-400" />
                    <span>Backend offline</span>
                  </>
                ) : (
                  <>
                    <span
                      className={`w-2 h-2 rounded-full ${
                        health.ckpt_exists ? "bg-green-400" : "bg-yellow-400"
                      }`}
                    />
                    <span className="uppercase">{health.device === "cuda" ? "GPU" : "CPU"}</span>
                    <span className="text-white/60">•</span>
                    <span>{health.ckpt_exists ? "model loaded" : "model missing"}</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
