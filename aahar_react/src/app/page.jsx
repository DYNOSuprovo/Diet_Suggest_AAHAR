"use client";
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { useUser } from '../context/UserContext';
import { Sparkles, Users, TrendingUp, Heart } from 'lucide-react';

// Typewriter effect component
const Typewriter = ({ words, className }) => {
    const [index, setIndex] = useState(0);
    const [text, setText] = useState('');
    const [isDeleting, setIsDeleting] = useState(false);

    useEffect(() => {
        const word = words[index];
        const timeout = setTimeout(() => {
            if (!isDeleting) {
                setText(word.substring(0, text.length + 1));
                if (text === word) {
                    setTimeout(() => setIsDeleting(true), 2000);
                }
            } else {
                setText(word.substring(0, text.length - 1));
                if (text === '') {
                    setIsDeleting(false);
                    setIndex((prev) => (prev + 1) % words.length);
                }
            }
        }, isDeleting ? 50 : 100);
        return () => clearTimeout(timeout);
    }, [text, isDeleting, index, words]);

    return <span className={className}>{text}<span style={{ opacity: 0.5 }}>|</span></span>;
};

export default function Home() {
    const { isOnboarded, isLoading } = useUser();
    const router = useRouter();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        if (!isLoading && isOnboarded) {
            router.push('/dashboard');
        }
    }, [isOnboarded, isLoading, router]);

    if (isLoading) {
        return (
            <div style={{
                height: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #1DB954 0%, #0d7a3a 100%)'
            }}>
                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    style={{
                        width: '50px',
                        height: '50px',
                        border: '4px solid rgba(255,255,255,0.3)',
                        borderTop: '4px solid white',
                        borderRadius: '50%'
                    }}
                />
            </div>
        );
    }

    if (isOnboarded) return null;

    const foodImages = [
        { emoji: '🍛', name: 'Curry', color: '#FFB74D' },
        { emoji: '🍚', name: 'Rice', color: '#E8F5E9' },
        { emoji: '🥗', name: 'Salad', color: '#A5D6A7' },
        { emoji: '🍲', name: 'Dal', color: '#FFCC80' },
        { emoji: '🫓', name: 'Roti', color: '#D7CCC8' },
        { emoji: '🥘', name: 'Biryani', color: '#FFAB91' }
    ];

    // Card hover animation variants
    const cardVariants = {
        initial: { opacity: 0, y: 30, rotateX: -15 },
        animate: (i) => ({
            opacity: 1,
            y: 0,
            rotateX: 0,
            transition: {
                delay: i * 0.1,
                duration: 0.6,
                type: "spring",
                stiffness: 100
            }
        }),
        hover: {
            scale: 1.08,
            rotateY: 5,
            rotateX: 5,
            boxShadow: "0 20px 40px rgba(0,0,0,0.15)",
            transition: { duration: 0.3 }
        },
        tap: { scale: 0.95 }
    };

    return (
        <div style={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            background: '#FAFAFA',
            overflow: 'hidden'
        }}>
            {/* Hero Section with Aahar in CENTER */}
            <div style={{
                background: 'linear-gradient(180deg, #E8F5E9 0%, #FAFAFA 100%)',
                padding: '16px',
                perspective: '1000px'
            }}>
                {/* 3x3 Grid with Aahar in the middle */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gridTemplateRows: 'repeat(3, 100px)',
                    gap: '10px'
                }}>
                    {/* Top Row - 3 food cards */}
                    {foodImages.slice(0, 3).map((food, i) => (
                        <motion.div
                            key={food.name}
                            custom={i}
                            variants={cardVariants}
                            initial="initial"
                            animate="animate"
                            whileHover="hover"
                            whileTap="tap"
                            style={{
                                background: food.color,
                                borderRadius: '20px',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: 'pointer',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
                            }}
                        >
                            <motion.span
                                animate={{
                                    y: [0, -6, 0],
                                    rotate: [0, 8, -8, 0]
                                }}
                                transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.2 }}
                                style={{ fontSize: '40px' }}
                            >
                                {food.emoji}
                            </motion.span>
                            <span style={{ fontSize: '11px', color: '#444', marginTop: '2px', fontWeight: '600' }}>
                                {food.name}
                            </span>
                        </motion.div>
                    ))}

                    {/* Middle Row - Food | AAHAR | Food */}
                    <motion.div
                        custom={3}
                        variants={cardVariants}
                        initial="initial"
                        animate="animate"
                        whileHover="hover"
                        whileTap="tap"
                        style={{
                            background: foodImages[3].color,
                            borderRadius: '20px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
                        }}
                    >
                        <motion.span
                            animate={{ y: [0, -6, 0], scale: [1, 1.1, 1] }}
                            transition={{ duration: 2, repeat: Infinity }}
                            style={{ fontSize: '40px' }}
                        >
                            {foodImages[3].emoji}
                        </motion.span>
                        <span style={{ fontSize: '11px', color: '#444', marginTop: '2px', fontWeight: '600' }}>
                            {foodImages[3].name}
                        </span>
                    </motion.div>

                    {/* CENTER - Aahar Logo */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0, rotate: 180 }}
                        animate={{ opacity: 1, scale: 1, rotate: 0 }}
                        transition={{
                            delay: 0.3,
                            duration: 0.8,
                            type: "spring",
                            stiffness: 100
                        }}
                        whileHover={{
                            scale: 1.08,
                            boxShadow: "0 20px 50px rgba(29, 185, 84, 0.5)",
                            rotate: 2
                        }}
                        whileTap={{ scale: 0.95 }}
                        style={{
                            background: '#1DB954',
                            borderRadius: '20px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            boxShadow: '0 8px 25px rgba(29, 185, 84, 0.4)',
                            color: 'white',
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                    >
                        {/* Pulsing glow effect */}
                        <motion.div
                            animate={{
                                scale: [1, 1.2, 1],
                                opacity: [0.3, 0.6, 0.3]
                            }}
                            transition={{ duration: 2, repeat: Infinity }}
                            style={{
                                position: 'absolute',
                                inset: 0,
                                background: 'radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%)',
                                borderRadius: '20px'
                            }}
                        />
                        <motion.h1
                            animate={{ scale: [1, 1.02, 1] }}
                            transition={{ duration: 3, repeat: Infinity }}
                            style={{
                                fontSize: '28px',
                                fontWeight: '800',
                                fontFamily: 'serif',
                                letterSpacing: '2px',
                                margin: 0,
                                zIndex: 1
                            }}
                        >
                            Aahar
                        </motion.h1>
                        <motion.p
                            style={{
                                fontSize: '8px',
                                opacity: 0.9,
                                letterSpacing: '1px',
                                margin: '4px 0 0 0',
                                zIndex: 1
                            }}
                        >
                            Nutrition • Health
                        </motion.p>
                    </motion.div>

                    <motion.div
                        custom={4}
                        variants={cardVariants}
                        initial="initial"
                        animate="animate"
                        whileHover="hover"
                        whileTap="tap"
                        style={{
                            background: foodImages[4].color,
                            borderRadius: '20px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
                        }}
                    >
                        <motion.span
                            animate={{ y: [0, -6, 0], rotate: [0, -5, 5, 0] }}
                            transition={{ duration: 2.5, repeat: Infinity }}
                            style={{ fontSize: '40px' }}
                        >
                            {foodImages[4].emoji}
                        </motion.span>
                        <span style={{ fontSize: '11px', color: '#444', marginTop: '2px', fontWeight: '600' }}>
                            {foodImages[4].name}
                        </span>
                    </motion.div>

                    {/* Bottom Row - 3 food cards (using last food item repeated) */}
                    <motion.div
                        custom={5}
                        variants={cardVariants}
                        initial="initial"
                        animate="animate"
                        whileHover="hover"
                        whileTap="tap"
                        style={{
                            background: foodImages[5].color,
                            borderRadius: '20px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
                        }}
                    >
                        <motion.span
                            animate={{ y: [0, -6, 0], scale: [1, 1.15, 1] }}
                            transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                            style={{ fontSize: '40px' }}
                        >
                            {foodImages[5].emoji}
                        </motion.span>
                        <span style={{ fontSize: '11px', color: '#444', marginTop: '2px', fontWeight: '600' }}>
                            {foodImages[5].name}
                        </span>
                    </motion.div>

                    <motion.div
                        custom={6}
                        variants={cardVariants}
                        initial="initial"
                        animate="animate"
                        whileHover="hover"
                        whileTap="tap"
                        style={{
                            background: '#B2DFDB',
                            borderRadius: '20px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
                        }}
                    >
                        <motion.span
                            animate={{ y: [0, -6, 0], scale: [1, 1.15, 1] }}
                            transition={{ duration: 2.2, repeat: Infinity, delay: 0.6 }}
                            style={{ fontSize: '40px' }}
                        >
                            🥣
                        </motion.span>
                        <span style={{ fontSize: '11px', color: '#444', marginTop: '2px', fontWeight: '600' }}>
                            Soup
                        </span>
                    </motion.div>

                    <motion.div
                        custom={7}
                        variants={cardVariants}
                        initial="initial"
                        animate="animate"
                        whileHover="hover"
                        whileTap="tap"
                        style={{
                            background: '#FFCDD2',
                            borderRadius: '20px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
                        }}
                    >
                        <motion.span
                            animate={{ y: [0, -6, 0], scale: [1, 1.15, 1] }}
                            transition={{ duration: 2.4, repeat: Infinity, delay: 0.7 }}
                            style={{ fontSize: '40px' }}
                        >
                            🍜
                        </motion.span>
                        <span style={{ fontSize: '11px', color: '#444', marginTop: '2px', fontWeight: '600' }}>
                            Noodles
                        </span>
                    </motion.div>
                </div>
            </div>

            {/* Bottom Section */}
            <motion.div
                initial={{ y: 100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.6, type: "spring", stiffness: 100 }}
                style={{
                    flex: 1,
                    background: 'white',
                    borderTopLeftRadius: '40px',
                    borderTopRightRadius: '40px',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    padding: '40px 24px 80px 24px',
                    textAlign: 'center',
                    boxShadow: '0 -10px 40px rgba(0,0,0,0.08)'
                }}
            >
                {/* Typewriter Tagline */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1.2 }}
                    style={{
                        fontSize: '26px',
                        fontWeight: '700',
                        color: '#1a1a1a',
                        marginBottom: '16px',
                        lineHeight: '1.3'
                    }}
                >
                    Let's start your<br />
                    <span style={{ color: '#1DB954' }}>
                        <Typewriter words={['health journey', 'fitness goals', 'nutrition tracking', 'better lifestyle']} />
                    </span>
                </motion.div>



                {/* Stats Pills */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 1.5 }}
                    style={{
                        display: 'flex',
                        gap: '12px',
                        marginBottom: '32px',
                        flexWrap: 'wrap',
                        justifyContent: 'center'
                    }}
                >
                    {[
                        { icon: TrendingUp, text: 'Track Calories', color: '#1DB954' },
                        { icon: Heart, text: 'Stay Healthy', color: '#E91E63' },
                        { icon: Sparkles, text: 'AI Powered', color: '#9C27B0' }
                    ].map((item, i) => (
                        <motion.div
                            key={i}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                padding: '10px 16px',
                                background: `${item.color}15`,
                                borderRadius: '20px',
                                color: item.color,
                                fontSize: '13px',
                                fontWeight: '600'
                            }}
                        >
                            <item.icon size={16} />
                            {item.text}
                        </motion.div>
                    ))}
                </motion.div>

                {/* CTA Button */}
                <Link href="/login" style={{ width: '100%', maxWidth: '320px', textDecoration: 'none' }}>
                    <motion.button
                        whileHover={{ scale: 1.02, boxShadow: '0 8px 30px rgba(29, 185, 84, 0.4)' }}
                        whileTap={{ scale: 0.98 }}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 1.6 }}
                        style={{
                            width: '100%',
                            padding: '18px 32px',
                            background: '#1DB954',
                            color: 'white',
                            border: 'none',
                            borderRadius: '16px',
                            fontSize: '18px',
                            fontWeight: '700',
                            cursor: 'pointer',
                            boxShadow: '0 4px 20px rgba(29, 185, 84, 0.3)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '8px'
                        }}
                    >
                        <Sparkles size={20} />
                        Get Started
                    </motion.button>
                </Link>


            </motion.div>
        </div>
    );
}
