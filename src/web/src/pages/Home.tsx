import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Camera, FileText, Workflow } from "lucide-react";
import Card from "../components/Card";

const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

export default function Home() {
  const options = [
    {
      icon: Camera,
      title: "Image Only",
      description:
        "Upload an ultrasound image for PCOS detection using computer vision.",
      path: "/image",
      color: "from-blue-500 to-teal-500",
    },
    {
      icon: FileText,
      title: "EHR Only",
      description:
        "Fill out medical history and lab values for risk assessment.",
      path: "/ehr",
      color: "from-purple-500 to-pink-500",
    },
    {
      icon: Workflow,
      title: "Hybrid",
      description:
        "Combine image and EHR data for comprehensive AI-driven diagnosis.",
      path: "/hybrid",
      color: "from-indigo-500 to-blue-500",
    },
  ];

  return (
    <div className="container mx-auto px-4 py-12">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-16"
      >
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-300 to-teal-300 bg-clip-text text-transparent">
          PCOS Detection Lab
        </h1>
        <p className="text-xl text-white/80 max-w-2xl mx-auto">
          Advanced AI-powered research platform for Polycystic Ovary Syndrome
          detection
        </p>
        <p className="text-sm text-white/60 mt-4 italic">
          Research prototype. Not for clinical use.
        </p>
      </motion.div>

      <motion.div
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto"
      >
        {options.map((option, index) => {
          const Icon = option.icon;
          return (
            <motion.div key={index} variants={fadeInUp}>
              <Card hover={true} className="h-full flex flex-col">
                <div className="flex-1">
                  <div
                    className={`w-16 h-16 rounded-xl bg-gradient-to-br ${option.color} flex items-center justify-center mb-4`}
                  >
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold mb-3">{option.title}</h2>
                  <p className="text-white/70 mb-6">{option.description}</p>
                </div>
                <Link to={option.path} className="block">
                  <button className="btn-primary w-full">
                    Start with {option.title}
                  </button>
                </Link>
              </Card>
            </motion.div>
          );
        })}
      </motion.div>

      <motion.div
        variants={fadeInUp}
        initial="initial"
        animate="animate"
        className="max-w-4xl mx-auto mt-16"
      >
        <Card>
          <div className="prose prose-invert max-w-none">
            <h3 className="text-2xl font-bold mb-4 text-white">
              About This Platform
            </h3>
            <div className="space-y-4 text-white/80">
              <p>
                This research-grade platform integrates state-of-the-art machine
                learning models to assist in PCOS detection through multiple
                modalities:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>
                  <strong>Image Analysis:</strong> Deep learning-based
                  ultrasound interpretation
                </li>
                <li>
                  <strong>Clinical Data:</strong> Evidence-based EHR scoring
                  with calibration
                </li>
                <li>
                  <strong>Multimodal Fusion:</strong> Combined AI predictions
                  for enhanced accuracy
                </li>
              </ul>
              <p className="text-sm text-white/60 mt-6 italic border-l-4 border-blue-500 pl-4">
                Important: This tool is for research and educational purposes
                only. It should not be used as a substitute for professional
                medical diagnosis or treatment.
              </p>
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  );
}
