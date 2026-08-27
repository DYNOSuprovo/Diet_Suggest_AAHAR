"use client";
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronLeft, User, Ruler, Activity, Check, Sparkles, Target } from 'lucide-react';
import { useUser } from '../../context/UserContext';

export default function Onboarding() {
    const router = useRouter();
    const { user, completeOnboarding, resetApp } = useUser();
    const [step, setStep] = useState(1);

    const [formData, setFormData] = useState({
        name: '',
        age: '',
        gender: 'male',
        height: '',
        weight: '',
        activityLevel: 'sedentary',
        customCalories: '',
        customWater: ''
    });

    useEffect(() => {
        if (user && user.name) {
            setFormData(prev => ({ ...prev, name: user.name }));
        }
    }, [user]);

    const totalSteps = 4;

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const validateStep = () => {
        if (step === 1) return formData.name && formData.name.trim().length > 0;
        if (step === 2) return formData.age > 0 && formData.height > 0 && formData.weight > 0;
        if (step === 3) return !!formData.activityLevel;
        if (step === 4) return true; // Goals step always valid
        return false;
    };

    const handleNext = () => {
        if (!validateStep()) return;
        if (step < totalSteps) {
            setStep(step + 1);
        } else {
            handleSubmit();
        }
    };

    const handleBack = () => {
        if (step > 1) setStep(step - 1);
    };

    const getCalculatedGoals = () => {
        const weight = parseFloat(formData.weight) || 0;
        const height = parseFloat(formData.height) || 0;
        const age = parseFloat(formData.age) || 0;

        let bmr = 10 * weight + 6.25 * height - 5 * age;
        bmr += formData.gender === 'male' ? 5 : -161;

        const activityMultipliers = {
            sedentary: 1.2,
            light: 1.375,
            moderate: 1.55,
            active: 1.725,
            very_active: 1.9
        };

        const tdee = Math.round(bmr * (activityMultipliers[formData.activityLevel] || 1.2));
        const water = Math.round(weight * 35);
        const heightInM = height / 100;
        const bmi = (weight > 0 && height > 0) ? (weight / (heightInM * heightInM)).toFixed(1) : 0;

        return { tdee: tdee > 0 ? tdee : 2000, water: water > 0 ? water : 2000, bmi };
    };

    const calculatedGoals = getCalculatedGoals();

    const calculateGoals = () => {
        const weight = parseFloat(formData.weight) || 0;
        const height = parseFloat(formData.height) || 0;
        const age = parseFloat(formData.age) || 0;

        // Use custom values if provided, otherwise use calculated
        const finalCalories = formData.customCalories ? parseInt(formData.customCalories) : calculatedGoals.tdee;
        const finalWater = formData.customWater ? parseInt(formData.customWater) : calculatedGoals.water;

        return {
            ...user,
            ...formData,
            weight, height, age,
            bmi: calculatedGoals.bmi,
            goalCalories: finalCalories,
            goalWater: finalWater
        };
    };

    const handleSubmit = () => {
        const finalData = calculateGoals();
        completeOnboarding(finalData);
        router.push('/dashboard');
    };

    const isValid = validateStep();

    const stepTitles = [
        { title: "👋 Let's get to know you!", desc: "Tell us your name to personalize your experience" },
        { title: "📊 Your Body Stats", desc: "This helps us calculate your calorie and water needs" },
        { title: "🏃 Activity Level", desc: "How active are you on a daily basis?" },
        { title: "🎯 Your Daily Goals", desc: "Based on your stats, here are your recommended targets" }
    ];

    const activityOptions = [
        { value: 'sedentary', label: 'Sedentary', desc: 'Little or no exercise', emoji: '🪑' },
        { value: 'light', label: 'Lightly Active', desc: 'Exercise 1-3 times/week', emoji: '🚶' },
        { value: 'moderate', label: 'Moderately Active', desc: 'Exercise 4-5 times/week', emoji: '🏃' },
        { value: 'active', label: 'Very Active', desc: 'Daily exercise or physical job', emoji: '💪' }
    ];

    return (
        <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #f0fdf4 0%, #ecfeff 50%, #f8fafc 100%)',
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            position: 'relative'
        }}>
            {/* Decorative Elements */}
            <div style={{ position: 'absolute', top: '10%', right: '5%', width: '100px', height: '100px', borderRadius: '50%', background: 'rgba(29, 185, 84, 0.1)', filter: 'blur(40px)' }} />
            <div style={{ position: 'absolute', bottom: '20%', left: '5%', width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(59, 130, 246, 0.1)', filter: 'blur(30px)' }} />

            {/* Restart Button */}
            <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={() => {
                    if (confirm('Start over? This will clear your progress.')) {
                        resetApp();
                        router.push('/');
                    }
                }}
                style={{
                    position: 'absolute', top: '20px', right: '20px',
                    background: 'rgba(239, 68, 68, 0.1)', border: 'none',
                    color: '#EF4444', fontSize: '12px', fontWeight: '600',
                    cursor: 'pointer', zIndex: 10, padding: '8px 16px',
                    borderRadius: '10px'
                }}
            >
                Restart
            </motion.button>

            {/* Progress Bar */}
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
                style={{ marginBottom: '30px', marginTop: '50px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                    <span style={{ color: '#1DB954', fontWeight: '700', fontSize: '14px' }}>Step {step} of {totalSteps}</span>
                    <span style={{ color: '#64748b', fontWeight: '600', fontSize: '14px' }}>{Math.round((step / totalSteps) * 100)}%</span>
                </div>
                <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '10px', overflow: 'hidden' }}>
                    <motion.div
                        animate={{ width: `${(step / totalSteps) * 100}%` }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                        style={{ height: '100%', background: 'linear-gradient(90deg, #1DB954, #22c55e)', borderRadius: '10px' }}
                    />
                </div>
            </motion.div>

            {/* Content */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <AnimatePresence mode="wait">
                    <motion.div
                        key={step}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                    >
                        {/* Step Header */}
                        <h1 style={{ fontSize: '28px', fontWeight: '800', color: '#0f172a', marginBottom: '8px' }}>
                            {stepTitles[step - 1].title}
                        </h1>
                        <p style={{ color: '#64748b', marginBottom: '30px', fontSize: '15px' }}>
                            {stepTitles[step - 1].desc}
                        </p>

                        {step === 1 && (
                            <div>
                                <label style={{ display: 'block', marginBottom: '10px', fontWeight: '600', color: '#334155', fontSize: '14px' }}>Your Name</label>
                                <div style={{ position: 'relative' }}>
                                    <User size={20} color="#94a3b8" style={{ position: 'absolute', left: '16px', top: '18px' }} />
                                    <input
                                        type="text" name="name" value={formData.name} onChange={handleInputChange}
                                        placeholder="Enter your name"
                                        style={{
                                            width: '100%', padding: '18px 18px 18px 50px', borderRadius: '16px',
                                            border: '2px solid #e2e8f0', fontSize: '16px', outline: 'none',
                                            background: 'white', color: '#0f172a', transition: 'border-color 0.2s'
                                        }}
                                        onFocus={(e) => e.target.style.borderColor = '#1DB954'}
                                        onBlur={(e) => e.target.style.borderColor = '#e2e8f0'}
                                    />
                                </div>
                            </div>
                        )}

                        {step === 2 && (
                            <div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                                    <div>
                                        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#334155', fontSize: '14px' }}>Age</label>
                                        <input
                                            type="number" name="age" value={formData.age} onChange={handleInputChange} placeholder="25"
                                            style={{
                                                width: '100%', padding: '16px', borderRadius: '14px',
                                                border: '2px solid #e2e8f0', fontSize: '16px', outline: 'none', background: 'white', color: '#0f172a'
                                            }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#334155', fontSize: '14px' }}>Gender</label>
                                        <select
                                            name="gender" value={formData.gender} onChange={handleInputChange}
                                            style={{
                                                width: '100%', padding: '16px', borderRadius: '14px',
                                                border: '2px solid #e2e8f0', fontSize: '16px', outline: 'none', background: 'white', color: '#0f172a'
                                            }}
                                        >
                                            <option value="male">Male</option>
                                            <option value="female">Female</option>
                                            <option value="other">Other</option>
                                        </select>
                                    </div>
                                </div>

                                <div style={{ marginBottom: '20px' }}>
                                    <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#334155', fontSize: '14px' }}>Height (cm)</label>
                                    <div style={{ position: 'relative' }}>
                                        <Ruler size={20} color="#94a3b8" style={{ position: 'absolute', left: '16px', top: '18px' }} />
                                        <input
                                            type="number" name="height" value={formData.height} onChange={handleInputChange} placeholder="170"
                                            style={{
                                                width: '100%', padding: '18px 18px 18px 50px', borderRadius: '14px',
                                                border: '2px solid #e2e8f0', fontSize: '16px', outline: 'none', background: 'white', color: '#0f172a'
                                            }}
                                        />
                                    </div>
                                </div>

                                <div>
                                    <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#334155', fontSize: '14px' }}>Weight (kg)</label>
                                    <div style={{ position: 'relative' }}>
                                        <Activity size={20} color="#94a3b8" style={{ position: 'absolute', left: '16px', top: '18px' }} />
                                        <input
                                            type="number" name="weight" value={formData.weight} onChange={handleInputChange} placeholder="65"
                                            style={{
                                                width: '100%', padding: '18px 18px 18px 50px', borderRadius: '14px',
                                                border: '2px solid #e2e8f0', fontSize: '16px', outline: 'none', background: 'white', color: '#0f172a'
                                            }}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}

                        {step === 3 && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                {activityOptions.map((option) => (
                                    <motion.div
                                        key={option.value}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={() => setFormData(prev => ({ ...prev, activityLevel: option.value }))}
                                        style={{
                                            padding: '18px 20px', borderRadius: '16px',
                                            border: formData.activityLevel === option.value ? '2px solid #1DB954' : '2px solid #e2e8f0',
                                            background: formData.activityLevel === option.value ? 'linear-gradient(135deg, #f0fdf4, #dcfce7)' : 'white',
                                            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                            boxShadow: formData.activityLevel === option.value ? '0 4px 15px rgba(29,185,84,0.15)' : '0 2px 8px rgba(0,0,0,0.04)',
                                            transition: 'all 0.2s'
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                                            <span style={{ fontSize: '28px' }}>{option.emoji}</span>
                                            <div>
                                                <div style={{ fontWeight: '700', color: '#0f172a', fontSize: '15px' }}>{option.label}</div>
                                                <div style={{ fontSize: '13px', color: '#64748b' }}>{option.desc}</div>
                                            </div>
                                        </div>
                                        {formData.activityLevel === option.value && (
                                            <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}
                                                style={{ width: '24px', height: '24px', borderRadius: '50%', background: '#1DB954', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                <Check size={14} color="white" />
                                            </motion.div>
                                        )}
                                    </motion.div>
                                ))}
                            </div>
                        )}

                        {step === 4 && (
                            <div>
                                {/* Calculated Goals Display */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
                                    {/* Calorie Goal */}
                                    <div style={{
                                        background: 'linear-gradient(135deg, #FFF7ED, #FFEDD5)',
                                        borderRadius: '20px', padding: '20px', textAlign: 'center',
                                        border: '2px solid #FB923C'
                                    }}>
                                        <div style={{ fontSize: '32px', marginBottom: '8px' }}>🔥</div>
                                        <div style={{ fontSize: '28px', fontWeight: '800', color: '#EA580C' }}>
                                            {formData.customCalories || calculatedGoals.tdee}
                                        </div>
                                        <div style={{ fontSize: '12px', color: '#9A3412', fontWeight: '600' }}>kcal/day</div>
                                    </div>

                                    {/* Water Goal */}
                                    <div style={{
                                        background: 'linear-gradient(135deg, #EFF6FF, #DBEAFE)',
                                        borderRadius: '20px', padding: '20px', textAlign: 'center',
                                        border: '2px solid #3B82F6'
                                    }}>
                                        <div style={{ fontSize: '32px', marginBottom: '8px' }}>💧</div>
                                        <div style={{ fontSize: '28px', fontWeight: '800', color: '#1D4ED8' }}>
                                            {formData.customWater || calculatedGoals.water}
                                        </div>
                                        <div style={{ fontSize: '12px', color: '#1E40AF', fontWeight: '600' }}>ml/day</div>
                                    </div>
                                </div>

                                {/* Customize Goals */}
                                <div style={{
                                    background: 'white', borderRadius: '16px', padding: '20px',
                                    border: '2px solid #e2e8f0', marginBottom: '16px'
                                }}>
                                    <div style={{ fontSize: '14px', fontWeight: '700', color: '#0f172a', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Target size={18} color="#1DB954" />
                                        Customize Your Goals (Optional)
                                    </div>

                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                        <div>
                                            <label style={{ fontSize: '12px', color: '#64748b', marginBottom: '6px', display: 'block' }}>Calorie Goal</label>
                                            <input
                                                type="number"
                                                name="customCalories"
                                                value={formData.customCalories}
                                                onChange={handleInputChange}
                                                placeholder={calculatedGoals.tdee.toString()}
                                                style={{
                                                    width: '100%', padding: '12px', borderRadius: '12px',
                                                    border: '2px solid #e2e8f0', fontSize: '16px', outline: 'none',
                                                    textAlign: 'center', fontWeight: '700',
                                                    background: '#f1f5f9', color: '#0f172a'
                                                }}
                                            />
                                        </div>
                                        <div>
                                            <label style={{ fontSize: '12px', color: '#64748b', marginBottom: '6px', display: 'block' }}>Water Goal (ml)</label>
                                            <input
                                                type="number"
                                                name="customWater"
                                                value={formData.customWater}
                                                onChange={handleInputChange}
                                                placeholder={calculatedGoals.water.toString()}
                                                style={{
                                                    width: '100%', padding: '12px', borderRadius: '12px',
                                                    border: '2px solid #e2e8f0', fontSize: '16px', outline: 'none',
                                                    textAlign: 'center', fontWeight: '700',
                                                    background: '#f1f5f9', color: '#0f172a'
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div style={{ background: '#F0FDF4', borderRadius: '12px', padding: '14px', display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                                    <Sparkles size={18} color="#16a34a" style={{ marginTop: '2px', flexShrink: 0 }} />
                                    <p style={{ fontSize: '13px', color: '#166534', lineHeight: '1.5' }}>
                                        These goals are calculated using the Mifflin-St Jeor equation. Feel free to adjust based on your personal needs!
                                    </p>
                                </div>
                            </div>
                        )}
                    </motion.div>
                </AnimatePresence>
            </div>

            {/* Navigation Buttons */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
                style={{ display: 'flex', gap: '14px', marginTop: '20px' }}>
                {step > 1 && (
                    <motion.button
                        whileTap={{ scale: 0.95 }}
                        onClick={handleBack}
                        style={{
                            padding: '18px', borderRadius: '16px', border: 'none',
                            background: '#f1f5f9', color: '#64748b', fontWeight: '600',
                            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}
                    >
                        <ChevronLeft size={24} />
                    </motion.button>
                )}
                <motion.button
                    whileTap={{ scale: 0.98 }}
                    onClick={handleNext}
                    disabled={!isValid}
                    style={{
                        flex: 1, padding: '18px', borderRadius: '16px', border: 'none',
                        background: isValid ? 'linear-gradient(135deg, #1DB954, #16a34a)' : '#e2e8f0',
                        color: isValid ? 'white' : '#94a3b8',
                        fontWeight: '700', fontSize: '16px',
                        cursor: isValid ? 'pointer' : 'not-allowed',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
                        boxShadow: isValid ? '0 8px 25px rgba(29, 185, 84, 0.35)' : 'none',
                        transition: 'all 0.3s'
                    }}
                >
                    {step === totalSteps ? (
                        <>
                            <Sparkles size={20} />
                            Complete Setup
                        </>
                    ) : (
                        <>
                            Continue
                            <ChevronRight size={20} />
                        </>
                    )}
                </motion.button>
            </motion.div>
        </div>
    );
}
