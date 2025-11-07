import { useRef, useState, DragEvent } from "react";
import { Upload, Image as ImageIcon, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface UploadBoxProps {
  onFile: (file: File | null) => void;
  accept?: string;
  currentFile?: File | null;
}

export default function UploadBox({
  onFile,
  accept = "image/*",
  currentFile,
}: UploadBoxProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      onFile(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFile(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFile(null);
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`
        relative w-full border-2 border-dashed rounded-xl p-8 cursor-pointer
        transition-all duration-300
        ${
          isDragging
            ? "border-blue-400 bg-blue-500/10"
            : "border-white/20 hover:border-white/40"
        }
        ${currentFile ? "border-solid border-green-500/50" : ""}
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileSelect}
        className="hidden"
        aria-label="Upload image"
      />

      <AnimatePresence mode="wait">
        {currentFile ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative"
          >
            <img
              src={URL.createObjectURL(currentFile)}
              alt="Preview"
              className="w-full h-64 object-contain rounded-lg"
            />
            <button
              onClick={handleRemove}
              className="absolute top-2 right-2 p-2 bg-red-500 rounded-full hover:bg-red-600 transition-colors"
              aria-label="Remove image"
            >
              <X className="w-5 h-5" />
            </button>
            <p className="mt-2 text-center text-sm text-white/80 truncate">
              {currentFile.name}
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center text-center"
          >
            <ImageIcon className="w-16 h-16 mb-4 text-white/40" />
            <p className="text-lg font-semibold mb-2">Drop an image here</p>
            <p className="text-sm text-white/60 mb-4">or click to browse</p>
            <p className="text-xs text-white/40">Supports: PNG, JPG, JPEG</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
