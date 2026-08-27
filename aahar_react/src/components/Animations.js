"use client";
import { motion } from 'framer-motion';

// Slide up animation for content sections
export function SlideUp({ children, delay = 0 }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
                duration: 0.4,
                delay,
                ease: [0.4, 0, 0.2, 1]
            }}
        >
            {children}
        </motion.div>
    );
}

// Fade in animation
export function FadeIn({ children, delay = 0 }) {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay }}
        >
            {children}
        </motion.div>
    );
}

// Scale in animation (for cards)
export function ScaleIn({ children, delay = 0 }) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
                duration: 0.3,
                delay,
                ease: [0.34, 1.56, 0.64, 1]
            }}
        >
            {children}
        </motion.div>
    );
}

// Stagger container for lists
export function StaggerContainer({ children, staggerDelay = 0.1 }) {
    return (
        <motion.div
            initial="hidden"
            animate="visible"
            variants={{
                visible: {
                    transition: {
                        staggerChildren: staggerDelay
                    }
                }
            }}
        >
            {children}
        </motion.div>
    );
}

// Stagger item for individual list items
export function StaggerItem({ children }) {
    return (
        <motion.div
            variants={{
                hidden: { opacity: 0, y: 20 },
                visible: { opacity: 1, y: 0 }
            }}
            transition={{ duration: 0.3 }}
        >
            {children}
        </motion.div>
    );
}

// Slide from left
export function SlideFromLeft({ children, delay = 0 }) {
    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{
                duration: 0.4,
                delay,
                ease: [0.4, 0, 0.2, 1]
            }}
        >
            {children}
        </motion.div>
    );
}

// Slide from right
export function SlideFromRight({ children, delay = 0 }) {
    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{
                duration: 0.4,
                delay,
                ease: [0.4, 0, 0.2, 1]
            }}
        >
            {children}
        </motion.div>
    );
}

// Pop animation (for buttons, icons)
export function Pop({ children, delay = 0 }) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
                duration: 0.3,
                delay,
                type: 'spring',
                stiffness: 400,
                damping: 20
            }}
        >
            {children}
        </motion.div>
    );
}
