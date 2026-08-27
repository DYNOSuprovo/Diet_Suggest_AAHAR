"use client";
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { User, ChevronRight, Scale, Activity, Flame, Droplets, Edit2, Save, X, Camera, Award, TrendingUp, Shield, Info, LogOut, Moon, Sun, Zap, Ruler } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import { useUser } from '../../context/UserContext';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';
import { useLanguage } from '../../context/LanguageContext';
import SettingsPanel from '../../components/SettingsPanel';
import BottomNav from '../../components/BottomNav';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

// Get Gravatar URL from email
// Get Gravatar URL from email
const getGravatarUrl = (email, size = 200) => {
    // Crypto module is not available in client-side browser/webview
    // Returning null for now to prevent crash. 
    // TODO: Use a client-side MD5 library if Gravatar is needed.
    return null;
};

// Dynamic achievements based on user data
const getAchievements = (dailyStats, water, meals) => {
    const totalDays = dailyStats ? Object.keys(dailyStats).length : 0;
    return [
        { id: 'firstDay', name: 'First Step', icon: '🌱', earned: totalDays >= 1, description: 'Track your first day' },
        { id: 'streak3', name: '3 Day Streak', icon: '🔥', earned: totalDays >= 3, description: 'Use the app for 3 days' },
        { id: 'streak7', name: 'Week Warrior', icon: '⚡', earned: totalDays >= 7, description: 'Use the app for 7 days' },
        { id: 'hydration', name: 'Hydration Hero', icon: '💧', earned: water && water.current >= (water.goal || 2000), description: 'Reach water goal' },
    ];
};

const getBMICategory = (bmi) => {
    if (bmi < 18.5) return { label: 'Underweight', color: '#3B82F6' };
    if (bmi < 25) return { label: 'Healthy', color: '#10B981' };
    if (bmi < 30) return { label: 'Overweight', color: '#F59E0B' };
    return { label: 'Obese', color: '#EF4444' };
};

export default function Profile() {
    const router = useRouter();
    const { user, updateProfile, resetApp, dailyStats, water, meals } = useUser();
    const { currentUser, logout } = useAuth();
    const { t } = useLanguage();
    const { isDark } = useTheme();
    const [isEditing, setIsEditing] = useState(false);
    const [formData, setFormData] = useState({ ...user });
    const [imageError, setImageError] = useState(false);

    // Profile Pic
    const profilePicUrl = currentUser?.photoURL || (user.email ? getGravatarUrl(user.email) : null);
    const achievements = getAchievements(dailyStats, water, meals);

    useEffect(() => { setFormData({ ...user }); }, [user]);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: name === 'email' || name === 'name' ? value : Number(value)
        }));
    };

    const handleSave = () => {
        Object.keys(formData).forEach(key => updateProfile(key, formData[key]));
        if (formData.weight && formData.height) {
            const heightInM = formData.height / 100;
            updateProfile('bmi', (formData.weight / (heightInM * heightInM)).toFixed(1));
        }
        setIsEditing(false);
    };

    const handleCancel = () => { setFormData({ ...user }); setIsEditing(false); };

    // Chart Data
    const getChartData = () => {
        if (!dailyStats || Object.keys(dailyStats).length === 0) return { labels: ['Current'], data: [formData.weight] };
        const dates = Object.keys(dailyStats).sort().slice(-7);
        return {
            labels: dates.map(d => new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })),
            data: dates.map(d => dailyStats[d].weight || 0)
        };
    };
    const chartData = getChartData();

    const calculateBMI = (weight, height) => {
        if (!weight || !height || height === 0) return 0;
        const heightInM = height / 100;
        return weight / (heightInM * heightInM);
    };
    const bmi = parseFloat(user.bmi) || calculateBMI(formData.weight, formData.height);
    const bmiInfo = getBMICategory(bmi || 0);

    const data = {
        labels: chartData.labels,
        datasets: [{
            fill: true,
            label: t('profile_weight'),
            data: chartData.data,
            borderColor: '#1DB954',
            backgroundColor: 'rgba(29, 185, 84, 0.1)',
            tension: 0.4
        }]
    };

    const options = {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
            y: { grid: { display: false }, ticks: { color: isDark ? '#9CA3AF' : '#4B5563' } },
            x: { grid: { display: false }, ticks: { color: isDark ? '#9CA3AF' : '#4B5563' } }
        }
    };

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
                <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
                    style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <h1 style={{ fontSize: '26px', fontWeight: '800', color: 'var(--text-primary)' }}>{t('profile_title')}</h1>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        {isEditing ? (
                            <>
                                <motion.button whileTap={{ scale: 0.9 }} onClick={handleCancel}
                                    style={{ padding: '10px', borderRadius: '12px', background: '#FEE2E2', color: '#DC2626', border: 'none', cursor: 'pointer' }}>
                                    <X size={20} />
                                </motion.button>
                                <motion.button whileTap={{ scale: 0.9 }} onClick={handleSave}
                                    style={{ padding: '10px', borderRadius: '12px', background: '#D1FAE5', color: '#059669', border: 'none', cursor: 'pointer' }}>
                                    <Save size={20} />
                                </motion.button>
                            </>
                        ) : (
                            <motion.button whileTap={{ scale: 0.9 }} onClick={() => setIsEditing(true)}
                                style={{
                                    padding: '10px', borderRadius: '12px', background: 'var(--bg-card)', color: 'var(--text-primary)',
                                    border: '1px solid var(--border-color)', cursor: 'pointer', boxShadow: 'var(--shadow-sm)',
                                    backdropFilter: 'blur(12px)'
                                }}>
                                <Edit2 size={20} />
                            </motion.button>
                        )}
                    </div>
                </motion.div>

                {/* Profile Card */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                    style={{
                        background: 'var(--bg-gradient-header)',
                        borderRadius: '24px', padding: '3px', marginBottom: '20px',
                        boxShadow: '0 10px 40px rgba(29, 185, 84, 0.2)'
                    }}>
                    <div style={{
                        background: 'var(--bg-card)',
                        backdropFilter: 'blur(20px) saturate(200%)',
                        borderRadius: '22px',
                        padding: '24px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '20px'
                    }}>
                        <div style={{ position: 'relative' }}>
                            <div style={{
                                width: '90px', height: '90px', borderRadius: '20px',
                                background: 'linear-gradient(135deg, #f0fdf4, #dcfce7)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                overflow: 'hidden', border: '2px solid #fff', boxShadow: '0 8px 20px rgba(0,0,0,0.1)'
                            }}>
                                {profilePicUrl && !imageError ? (
                                    <img
                                        src={profilePicUrl}
                                        alt="Profile"
                                        onError={() => setImageError(true)}
                                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                    />
                                ) : (
                                    <User size={40} color="#15803d" />
                                )}
                            </div>
                        </div>
                        <div>
                            {isEditing ? (
                                <input name="name" value={formData.name} onChange={handleInputChange}
                                    style={{ fontSize: '20px', fontWeight: 'bold', width: '100%', marginBottom: '4px', padding: '4px', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--input-bg)', color: 'var(--text-primary)' }} />
                            ) : (
                                <h2 style={{ fontSize: '20px', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px' }}>{user.name || t('onboarding_name')}</h2>
                            )}
                            <p style={{ fontSize: '14px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <Award size={14} color="#EAB308" /> {t('profile_days_tracked')}: {Object.keys(dailyStats || {}).length}
                            </p>
                        </div>
                    </div>
                </motion.div>

                {/* Stats Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '20px' }}>
                    <div style={{ background: 'var(--bg-card)', backdropFilter: 'blur(20px) saturate(200%)', padding: '16px', borderRadius: '20px', border: '1px solid var(--border-color)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{t('profile_weight')}</span>
                            <Scale size={16} color="#1DB954" />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
                            {isEditing ? (
                                <input name="weight" type="number" value={formData.weight} onChange={handleInputChange}
                                    style={{ fontSize: '18px', fontWeight: 'bold', width: '60px', padding: '2px', background: 'var(--input-bg)', color: 'var(--text-primary)' }} />
                            ) : (
                                <span style={{ fontSize: '22px', fontWeight: '800', color: 'var(--text-primary)' }}>{user.weight}</span>
                            )}
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>kg</span>
                        </div>
                    </div>

                    <div style={{ background: 'var(--bg-card)', backdropFilter: 'blur(20px) saturate(200%)', padding: '16px', borderRadius: '20px', border: '1px solid var(--border-color)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{t('profile_height')}</span>
                            <Ruler size={16} color="#3B82F6" />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
                            {isEditing ? (
                                <input name="height" type="number" value={formData.height} onChange={handleInputChange}
                                    style={{ fontSize: '18px', fontWeight: 'bold', width: '60px', padding: '2px', background: 'var(--input-bg)', color: 'var(--text-primary)' }} />
                            ) : (
                                <span style={{ fontSize: '22px', fontWeight: '800', color: 'var(--text-primary)' }}>{user.height}</span>
                            )}
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>cm</span>
                        </div>
                    </div>

                    <div style={{ background: 'var(--bg-card)', backdropFilter: 'blur(20px) saturate(200%)', padding: '16px', borderRadius: '20px', border: '1px solid var(--border-color)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{t('profile_bmi')}</span>
                            <Activity size={16} color={bmiInfo.color} />
                        </div>
                        <div>
                            <span style={{ fontSize: '22px', fontWeight: '800', color: 'var(--text-primary)' }}>{bmi.toFixed(1)}</span>
                            <span style={{ fontSize: '12px', color: bmiInfo.color, marginLeft: '6px', fontWeight: '600' }}>{bmiInfo.label}</span>
                        </div>
                    </div>

                    <div style={{ background: 'var(--bg-card)', backdropFilter: 'blur(20px) saturate(200%)', padding: '16px', borderRadius: '20px', border: '1px solid var(--border-color)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{t('profile_age')}</span>
                            <User size={16} color="#F59E0B" />
                        </div>
                        {isEditing ? (
                            <input name="age" type="number" value={formData.age} onChange={handleInputChange}
                                style={{ fontSize: '22px', fontWeight: '800', width: '100%', padding: '2px', background: 'var(--input-bg)', color: 'var(--text-primary)' }} />
                        ) : (
                            <span style={{ fontSize: '22px', fontWeight: '800', color: 'var(--text-primary)' }}>{user.age}</span>
                        )}
                    </div>
                </div>

                {/* Achievements */}
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Flame size={20} color="#F59E0B" /> {t('profile_achievements')}
                </h3>
                <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '16px', scrollbarWidth: 'none' }}>
                    {achievements.map(ach => (
                        <div key={ach.id} style={{
                            minWidth: '100px', padding: '12px', borderRadius: '16px',
                            background: ach.earned ? 'linear-gradient(135deg, #1DB954 0%, #10B981 100%)' : 'var(--bg-card)',
                            backdropFilter: ach.earned ? 'none' : 'blur(12px)',
                            border: ach.earned ? 'none' : '1px dashed var(--border-color)',
                            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '8px',
                            opacity: ach.earned ? 1 : 0.6
                        }}>
                            <span style={{ fontSize: '24px' }}>{ach.icon}</span>
                            <span style={{ fontSize: '11px', fontWeight: '600', color: ach.earned ? 'white' : 'var(--text-secondary)', textAlign: 'center' }}>{ach.name}</span>
                        </div>
                    ))}
                </div>

                {/* Weight Chart */}
                <div style={{ background: 'var(--bg-card)', backdropFilter: 'blur(20px) saturate(200%)', padding: '20px', borderRadius: '24px', marginBottom: '24px', border: '1px solid var(--border-color)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ fontSize: '16px', fontWeight: '600', color: 'var(--text-primary)' }}>{t('profile_stats')}</h3>
                        <TrendingUp size={18} color="#1DB954" />
                    </div>
                    <div style={{ height: '150px' }}>
                        <Line data={data} options={options} />
                    </div>
                </div>

                {/* Settings Panel */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }} style={{ marginBottom: '20px' }}>
                    <SettingsPanel />
                </motion.div>

                {/* Buttons */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} style={{ textAlign: 'center' }}>
                    {currentUser && (
                        <motion.button whileTap={{ scale: 0.95 }}
                            onClick={async () => { if (confirm('Logout?')) { await logout(); window.location.href = '/login'; } }}
                            style={{
                                color: 'white', background: 'linear-gradient(135deg, #1DB954, #16a34a)', border: 'none', padding: '14px 28px',
                                borderRadius: '14px', fontSize: '14px', fontWeight: '600', cursor: 'pointer',
                                display: 'flex', alignItems: 'center', gap: '8px', margin: '0 auto 12px',
                                boxShadow: '0 4px 15px rgba(29, 185, 84, 0.3)'
                            }}>
                            <LogOut size={18} /> {t('profile_logout')}
                        </motion.button>
                    )}
                    <motion.button whileTap={{ scale: 0.95 }}
                        onClick={async () => { if (confirm('Reset all data?')) { resetApp(); if (currentUser) await logout(); window.location.href = '/login'; } }}
                        style={{
                            color: '#EF4444', background: isDark ? 'rgba(239, 68, 68, 0.1)' : '#FEF2F2', border: 'none', padding: '14px 28px',
                            borderRadius: '14px', fontSize: '14px', fontWeight: '600', cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: '8px', margin: '0 auto'
                        }}>
                        <LogOut size={18} /> Reset & Logout
                    </motion.button>
                </motion.div>
            </div>
            <BottomNav />
        </>
    );
}
