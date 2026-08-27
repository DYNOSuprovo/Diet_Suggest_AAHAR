"use client";
import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GoalSuccessModal from '../../components/GoalSuccessModal';
import { Plus, Droplets, X, Award, Lightbulb, Clock, TrendingUp } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import { useUser } from '../../context/UserContext';
import { useLanguage } from '../../context/LanguageContext';
import { useTheme } from '../../context/ThemeContext';
import BottomNav from '../../components/BottomNav';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

// Quick add presets with icons
const quickAddOptions = [
    { amount: 250, label: 'Glass', icon: '🥛', color: '#E3F2FD' },
    { amount: 500, label: 'Bottle', icon: '🍶', color: '#E8F5E9' },
    { amount: 350, label: 'Cup', icon: '☕', color: '#FFF3E0' },
    { amount: 750, label: 'Large', icon: '🫗', color: '#F3E5F5' },
];

export default function Water() {
    const { user, water, addWater, removeWater, dailyStats, isLoading } = useUser();
    const { t } = useLanguage();
    const { isDark } = useTheme();
    const [customAmount, setCustomAmount] = useState('');
    const [isAdding, setIsAdding] = useState(false);
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    const [tipIndex, setTipIndex] = useState(0);

    const wasGoalReachedRef = useRef(false);
    const isFirstLoadRef = useRef(true);

    // Hydration tips
    const hydrationTips = [
        { tip: "Drink water first thing in the morning", icon: "🌅" },
        { tip: "Keep a water bottle at your desk", icon: "💼" },
        { tip: "Set hourly reminders to drink water", icon: "⏰" },
        { tip: "Drink before you feel thirsty", icon: "💧" },
        { tip: "Add lemon for flavor if plain water is boring", icon: "🍋" },
    ];

    // Rotate tips
    useEffect(() => {
        const interval = setInterval(() => setTipIndex(i => (i + 1) % hydrationTips.length), 8000);
        return () => clearInterval(interval);
    }, []);

    const effectiveGoal = user.goalWater > 0 ? user.goalWater : 2000;
    const maxWater = effectiveGoal * 1.5;
    const isMaxReached = water.current >= maxWater;
    const percentage = Math.min(100, Math.round((water.current / effectiveGoal) * 100));

    const handleAddWater = (amount) => {
        if (isAdding || isMaxReached) return;
        setIsAdding(true);
        addWater(amount);
        setTimeout(() => setIsAdding(false), 500);
    };

    const handleCustomAdd = () => {
        if (customAmount && !isNaN(customAmount)) {
            handleAddWater(parseInt(customAmount));
            setCustomAmount('');
        }
    };

    // Chart Data
    const getChartData = () => {
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const today = new Date();
        const labels = [], dataPoints = [];
        for (let i = 6; i >= 0; i--) {
            const d = new Date(today);
            d.setDate(today.getDate() - i);
            const dateStr = d.toISOString().split('T')[0];
            labels.push(i === 0 ? 'Today' : days[d.getDay()]);
            dataPoints.push(i === 0 ? water.current : (dailyStats?.[dateStr]?.water || 0));
        }
        return { labels, data: dataPoints };
    };

    const chartData = getChartData();
    const data = {
        labels: chartData.labels,
        datasets: [{
            label: 'Water (ml)',
            data: chartData.data,
            borderColor: '#2962FF',
            backgroundColor: 'rgba(41, 98, 255, 0.1)',
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#2962FF',
            pointBorderColor: isDark ? '#1e293b' : 'white',
            pointBorderWidth: 2,
            pointRadius: 5,
            pointHoverRadius: 7,
        }],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            y: {
                beginAtZero: true,
                grid: { borderDash: [4, 4], color: isDark ? '#334155' : '#E5E5E5' },
                ticks: { color: isDark ? '#94a3b8' : '#9CA3AF', font: { size: 10 } }
            },
            x: { grid: { display: false }, ticks: { color: isDark ? '#94a3b8' : '#9CA3AF', font: { size: 10 } } }
        }
    };

    useEffect(() => {
        if (isLoading) return;
        const isGoalReached = water.current >= effectiveGoal;
        if (isFirstLoadRef.current) {
            wasGoalReachedRef.current = isGoalReached;
            isFirstLoadRef.current = false;
            return;
        }
        if (!wasGoalReachedRef.current && isGoalReached && water.current > 0) setShowSuccessModal(true);
        wasGoalReachedRef.current = isGoalReached;
    }, [water.current, effectiveGoal, isLoading]);

    return (
        <>
            <div style={{
                padding: '20px',
                paddingBottom: '100px',
                background: 'var(--bg-gradient-main)',
                minHeight: '100vh',
                color: 'var(--text-primary)'
            }}>

                {/* Header */}
                <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: '20px' }}>
                    <h1 style={{ fontSize: '26px', fontWeight: '800', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        💧 {t('water_title')}
                    </h1>
                    <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Stay hydrated, stay healthy</p>
                </motion.div>

                {/* Main Card */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                    style={{
                        background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
                        borderRadius: '24px', padding: '28px 24px', color: 'white', textAlign: 'center',
                        marginBottom: '20px', boxShadow: '0 10px 40px rgba(29, 78, 216, 0.3)', position: 'relative', overflow: 'hidden'
                    }}>

                    {/* Water Tank */}
                    <div style={{ position: 'relative', width: '160px', height: '160px', margin: '0 auto 20px' }}>
                        <div style={{
                            position: 'absolute', inset: 0, borderRadius: '50%', border: '5px solid rgba(255,255,255,0.25)',
                            background: 'rgba(255,255,255,0.08)', overflow: 'hidden'
                        }}>
                            <motion.div
                                style={{ position: 'absolute', bottom: 0, left: 0, right: 0, background: 'linear-gradient(to top, rgba(255,255,255,0.6), rgba(255,255,255,0.35))' }}
                                initial={{ height: 0 }} animate={{ height: `${percentage}%` }} transition={{ duration: 0.8, ease: 'easeOut' }}
                            >
                                <motion.div
                                    style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '10px', background: 'linear-gradient(180deg, rgba(255,255,255,0.5), transparent)', borderRadius: '100% 100% 0 0' }}
                                    animate={{ y: [0, -3, 0] }} transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
                                />
                            </motion.div>
                        </div>
                        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', zIndex: 1 }}>
                            <motion.span key={water.current} initial={{ scale: 1.3 }} animate={{ scale: 1 }} style={{ fontSize: '36px', fontWeight: '800' }}>
                                {water.current}
                            </motion.span>
                            <span style={{ fontSize: '13px', opacity: 0.85 }}>of {effectiveGoal}ml</span>
                            <span style={{ fontSize: '18px', fontWeight: '700', marginTop: '4px' }}>{percentage}%</span>
                        </div>
                    </div>

                    {/* Quick Add Buttons */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '16px' }}>
                        {quickAddOptions.map(opt => (
                            <motion.button
                                key={opt.amount}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => handleAddWater(opt.amount)}
                                disabled={isAdding || isMaxReached}
                                style={{
                                    padding: '14px 8px', borderRadius: '14px', background: 'rgba(255,255,255,0.15)',
                                    color: 'white', border: '1px solid rgba(255,255,255,0.2)', cursor: isMaxReached ? 'not-allowed' : 'pointer',
                                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px',
                                    backdropFilter: 'blur(10px)', transition: 'all 0.2s', opacity: isMaxReached ? 0.5 : 1
                                }}
                            >
                                <span style={{ fontSize: '22px' }}>{opt.icon}</span>
                                <span style={{ fontSize: '12px', fontWeight: '600' }}>{opt.label}</span>
                                <span style={{ fontSize: '10px', opacity: 0.8 }}>{opt.amount}ml</span>
                            </motion.button>
                        ))}
                    </div>

                    {/* Custom Input */}
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            type="number" value={customAmount} onChange={(e) => setCustomAmount(e.target.value)}
                            placeholder="Custom ml..." onKeyDown={(e) => e.key === 'Enter' && handleCustomAdd()}
                            style={{
                                flex: 1, minWidth: 0, padding: '14px 12px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.2)',
                                background: 'rgba(255,255,255,0.15)', color: 'white', fontSize: '14px', outline: 'none'
                            }}
                        />
                        <motion.button whileTap={{ scale: 0.95 }} onClick={handleCustomAdd} disabled={isMaxReached}
                            style={{
                                padding: '14px 16px', borderRadius: '12px', background: 'white', color: '#1D4ED8',
                                border: 'none', fontWeight: '700', cursor: isMaxReached ? 'not-allowed' : 'pointer',
                                display: 'flex', alignItems: 'center', gap: '6px', opacity: isMaxReached ? 0.5 : 1,
                                flexShrink: 0
                            }}>
                            <Plus size={18} /> {t('water_add')}
                        </motion.button>
                    </div>
                </motion.div>

                {/* Hydration Tip */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
                    style={{ background: 'linear-gradient(135deg, #FEF3C7, #FDE68A)', borderRadius: '16px', padding: '16px', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '14px' }}>
                    <div style={{ fontSize: '28px' }}>{hydrationTips[tipIndex].icon}</div>
                    <div>
                        <div style={{ fontSize: '11px', fontWeight: '600', color: '#92400E', marginBottom: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Lightbulb size={12} /> HYDRATION TIP
                        </div>
                        <AnimatePresence mode="wait">
                            <motion.p key={tipIndex} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}
                                style={{ fontSize: '13px', color: '#78350F', fontWeight: '500' }}>{hydrationTips[tipIndex].tip}</motion.p>
                        </AnimatePresence>
                    </div>
                </motion.div>

                {/* Today's Log */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
                    style={{
                        background: 'var(--bg-card)',
                        backdropFilter: 'blur(20px) saturate(200%)',
                        borderRadius: '20px',
                        padding: '20px',
                        marginBottom: '20px',
                        boxShadow: 'var(--shadow-sm)',
                        border: '1px solid var(--border-color)'
                    }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                        <h3 style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Clock size={16} color="#3B82F6" /> {t('water_history')}
                        </h3>
                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{water.history.length} entries</span>
                    </div>
                    {water.history.length > 0 ? (
                        <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                            {water.history.map((entry, idx) => (
                                <motion.div key={idx} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }}
                                    style={{
                                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                        padding: '12px', background: 'var(--bg-secondary)', borderRadius: '12px', marginBottom: '8px'
                                    }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <div style={{
                                            width: '36px', height: '36px', borderRadius: '10px',
                                            background: 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center'
                                        }}>
                                            <Droplets size={18} color="white" />
                                        </div>
                                        <div>
                                            <div style={{ fontWeight: '600', fontSize: '14px', color: 'var(--text-primary)' }}>{entry.amount} ml</div>
                                            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{entry.time}</div>
                                        </div>
                                    </div>
                                    <button onClick={() => removeWater(idx)}
                                        style={{ background: 'none', border: 'none', color: '#cbd5e1', cursor: 'pointer', padding: '8px' }}>
                                        <X size={16} />
                                    </button>
                                </motion.div>
                            ))}
                        </div>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
                            <Droplets size={40} style={{ opacity: 0.3, marginBottom: '8px' }} />
                            <p style={{ fontSize: '13px' }}>No water logged yet</p>
                        </div>
                    )}
                </motion.div>

                {/* Weekly Chart */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
                    style={{
                        background: 'var(--bg-card)',
                        backdropFilter: 'blur(20px) saturate(200%)',
                        borderRadius: '20px',
                        padding: '20px',
                        boxShadow: 'var(--shadow-sm)',
                        border: '1px solid var(--border-color)'
                    }}>
                    <h3 style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <TrendingUp size={16} color="#10B981" /> Weekly Overview
                    </h3>
                    <div style={{ height: '200px' }}>
                        <Line data={data} options={options} />
                    </div>
                </motion.div>
            </div>

            <GoalSuccessModal isOpen={showSuccessModal} onClose={() => setShowSuccessModal(false)} />
            <BottomNav />
        </>
    );
}
