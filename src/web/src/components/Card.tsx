import { ReactNode } from "react";
import { motion } from "framer-motion";

interface CardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}

export default function Card({
  children,
  className = "",
  hover = false,
}: CardProps) {
  const cardContent = (
    <div className={`glass-card p-6 ${className}`}>{children}</div>
  );

  if (hover) {
    return (
      <motion.div whileHover={{ y: -4 }} transition={{ duration: 0.2 }}>
        {cardContent}
      </motion.div>
    );
  }

  return cardContent;
}
