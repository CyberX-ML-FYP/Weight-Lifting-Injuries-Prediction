import type { PropsWithChildren, CSSProperties } from "react";
import { motion } from "framer-motion";

interface GlassCardProps {
  className?: string;
  style?: CSSProperties;
  delay?: number;
}

/** Reusable rounded, glassmorphism-style card with a subtle entrance animation. */
export default function GlassCard({ children, className = "", style, delay = 0 }: PropsWithChildren<GlassCardProps>) {
  return (
    <motion.div
      className={`glass-card ${className}`}
      style={style}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}
