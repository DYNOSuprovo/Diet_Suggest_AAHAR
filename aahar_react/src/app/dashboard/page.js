"use client";
import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useUser } from '../../context/UserContext';
import { useLanguage } from '../../context/LanguageContext';
import { useTheme } from '../../context/ThemeContext';
import { Plus, Minus, Search, X, Utensils, Trash2, Flame, Zap, Target, Lightbulb } from 'lucide-react';
import { searchFood, analyzeMeal } from '../../lib/api';
import BottomNav from '../../components/BottomNav';

// Circular Progress Ring Component
const CalorieRing = ({ current, goal, size = 200 }) => {
    const { t } = useLanguage();
    const strokeWidth = 12;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const percentage = Math.min((current / goal) * 100, 100);
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    const getColor = () => {
        if (percentage >= 100) return '#ef4444';
        if (percentage >= 80) return '#f97316';
        return '#1DB954';
    };

    return (
        <div style={{ position: 'relative', width: size, height: size }}>
            <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
                {/* Background circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="var(--border-color)"
                    strokeWidth={strokeWidth}
                    opacity="0.3"
                />
                {/* Progress circle */}
                <motion.circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={getColor()}
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                />
            </svg>
            <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center'
            }}>
                <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.5, type: "spring" }}
                    style={{ fontSize: '36px', fontWeight: '800', color: 'var(--text-primary)' }}
                >
                    {current}
                </motion.div>
                <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>of {goal} cal</div>
                <div style={{
                    marginTop: '4px',
                    padding: '4px 12px',
                    background: `${getColor()}15`,
                    borderRadius: '12px',
                    color: getColor(),
                    fontSize: '12px',
                    fontWeight: '600'
                }}>
                    {Math.round(percentage)}%
                </div>
            </div>
        </div>
    );
};

export default function Dashboard() {
    const { user, meals, addFood, removeFood, preferences } = useUser();
    const { t } = useLanguage();
    const { isDark } = useTheme();
    const [isAddFoodOpen, setIsAddFoodOpen] = useState(false);
    const [selectedMealType, setSelectedMealType] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [selectedItemForQuant, setSelectedItemForQuant] = useState(null);
    const [itemQuantity, setItemQuantity] = useState(1);

    // Calculate totals
    const totalCalories = ['breakfast', 'lunch', 'snack', 'dinner'].reduce((acc, type) =>
        acc + meals[type].reduce((sum, item) => sum + parseInt(item.calories), 0), 0);

    const calculateTotal = (nutrient) => {
        const allMeals = [...meals.breakfast, ...meals.lunch, ...meals.snack, ...meals.dinner];
        return allMeals.reduce((acc, item) => acc + (parseFloat(item[nutrient]) || 0), 0);
    };

    const totalProtein = calculateTotal('protein');
    const totalCarbs = calculateTotal('carbs');
    const totalFat = calculateTotal('fat');
    const remainingCalories = user.goalCalories - totalCalories;

    // Greeting logic
    const getGreeting = () => {
        const hour = new Date().getHours();
        if (hour < 12) return { text: t('dashboard_greeting') || 'Good Morning', emoji: '🌅' };
        if (hour < 17) return { text: t('dashboard_greeting') || 'Good Afternoon', emoji: '☀️' };
        if (hour < 21) return { text: t('dashboard_greeting') || 'Good Evening', emoji: '🌆' };
        return { text: t('dashboard_greeting') || 'Good Night', emoji: '🌙' };
    };
    const greeting = getGreeting();

    const openAddFood = (mealType) => {
        setSelectedMealType(mealType);
        setIsAddFoodOpen(true);
        setSearchTerm('');
    };

    const handleFoodClick = (item) => {
        setSelectedItemForQuant(item);
        setItemQuantity(1);
    };

    const confirmAddFood = () => {
        if (!selectedItemForQuant) return;
        const multiplier = itemQuantity;
        const getVal = (key) => parseFloat(selectedItemForQuant[key]) || 0;

        addFood(selectedMealType, {
            name: selectedItemForQuant["Dish Name"],
            calories: Math.round(getVal("Calories (kcal)") * multiplier),
            protein: (getVal("Protein (g)") * multiplier).toFixed(1),
            carbs: (getVal("Carbs (g)") * multiplier).toFixed(1),
            fat: (getVal("Fat (g)") * multiplier).toFixed(1),
            quantity: multiplier,
            servingSize: selectedItemForQuant["Serving Size"],
        });

        setIsAddFoodOpen(false);
        setSelectedItemForQuant(null);
        setItemQuantity(1);
        setSearchTerm('');
    };

    // Filter by preferences
    const filterByPreferences = useCallback((results) => {
        if (!results || results.length === 0) return [];
        let filtered = results;
        const nonVegKeywords = ['chicken', 'mutton', 'fish', 'egg', 'prawn', 'meat', 'lamb', 'beef', 'pork'];
        const glutenKeywords = ['wheat', 'maida', 'atta', 'roti', 'naan', 'paratha', 'bread'];
        const dairyKeywords = ['paneer', 'cheese', 'milk', 'curd', 'yogurt', 'ghee', 'butter', 'cream'];

        if (preferences.vegetarian) {
            filtered = filtered.filter(item => {
                const dishName = item["Dish Name"]?.toLowerCase() || '';
                const category = item["Category"]?.toLowerCase() || '';
                return !nonVegKeywords.some(keyword => dishName.includes(keyword) || category.includes(keyword));
            });
        }
        if (preferences.glutenFree) filtered = filtered.filter(item => !glutenKeywords.some(keyword => (item["Dish Name"]?.toLowerCase() || '').includes(keyword)));
        if (preferences.dairyFree) filtered = filtered.filter(item => !dairyKeywords.some(keyword => (item["Dish Name"]?.toLowerCase() || '').includes(keyword)));

        return filtered;
    }, [preferences]);

    useEffect(() => {
        const delayDebounceFn = setTimeout(async () => {
            if (searchTerm.length > 2) {
                setIsSearching(true);
                const results = await searchFood(searchTerm);
                // Apply dietary preference filters
                const filteredResults = filterByPreferences(results);
                setSearchResults(filteredResults);
                setIsSearching(false);
            } else {
                setSearchResults([]);
            }
        }, 500);
        return () => clearTimeout(delayDebounceFn);
    }, [searchTerm, filterByPreferences]);

    {/* AI Analysis State - with localStorage persistence */ }
    const [analysis, setAnalysis] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // Load saved analysis on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem('aahar_analysis');
            if (saved) {
                setAnalysis(JSON.parse(saved));
            }
        } catch (e) {
            console.log('Could not load analysis');
        }
    }, []);

    // Save analysis when it changes
    useEffect(() => {
        if (analysis) {
            try {
                localStorage.setItem('aahar_analysis', JSON.stringify(analysis));
            } catch (e) {
                console.log('Could not save analysis');
            }
        }
    }, [analysis]);

    const handleAnalyze = async () => {
        setIsAnalyzing(true);
        setAnalysis(null);
        try {
            const allMeals = [...meals.breakfast, ...meals.lunch, ...meals.snack, ...meals.dinner];
            if (allMeals.length === 0) {
                setAnalysis({ type: 'warning', text: "Please add some food items first!" });
                setIsAnalyzing(false);
                return;
            }
            const dishNames = allMeals.map(m => m.name);
            const result = await analyzeMeal(dishNames);
            setAnalysis({ type: 'success', text: result.analysis });
        } catch (error) {
            setAnalysis({ type: 'error', text: "AI is currently unavailable. Try again later." });
        } finally {
            setIsAnalyzing(false);
        }
    };

    const MealSection = ({ titleKey, items, type }) => (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            style={{
                background: 'var(--bg-card)',
                backdropFilter: 'blur(20px) saturate(200%)',
                WebkitBackdropFilter: 'blur(12px)',
                borderRadius: '20px',
                padding: '20px',
                marginBottom: '16px',
                boxShadow: 'var(--shadow-sm)',
                border: '1px solid var(--border-color)'
            }}
        >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '24px' }}>{type === 'breakfast' ? '🍳' : type === 'lunch' ? '🍛' : type === 'snack' ? '🍿' : '🍲'}</span>
                    <h3 style={{ fontSize: '17px', fontWeight: '700', color: 'var(--text-primary)' }}>{t(titleKey)}</h3>
                </div>
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => openAddFood(type)}
                    style={{
                        background: '#1DB954',
                        border: 'none',
                        color: 'white',
                        padding: '8px 16px',
                        borderRadius: '20px',
                        fontWeight: '600',
                        fontSize: '13px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        cursor: 'pointer',
                        boxShadow: '0 4px 12px rgba(29, 185, 84, 0.25)'
                    }}
                >
                    <Plus size={16} /> {t('dashboard_add_food')}
                </motion.button>
            </div>

            {items.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {items.map((item, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            style={{
                                background: 'var(--bg-secondary)',
                                backdropFilter: 'blur(20px) saturate(200%)',
                                borderRadius: '14px',
                                padding: '14px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                border: '1px solid var(--border-light)'
                            }}
                        >
                            <div style={{
                                width: '44px',
                                height: '44px',
                                borderRadius: '12px',
                                background: 'linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%)',
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center'
                            }}>
                                <Utensils size={20} color="#1DB954" />
                            </div>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontWeight: '600', fontSize: '15px', color: 'var(--text-primary)' }}>{item.name}</div>
                                <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                                    {item.calories} cal
                                </div>
                            </div>
                            <motion.button
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                onClick={() => removeFood(type, idx)}
                                style={{
                                    background: 'rgba(239, 68, 68, 0.1)',
                                    border: 'none',
                                    borderRadius: '10px',
                                    padding: '8px',
                                    cursor: 'pointer',
                                    color: '#EF4444'
                                }}
                            >
                                <Trash2 size={16} />
                            </motion.button>
                        </motion.div>
                    ))}
                </div>
            ) : (
                <div style={{
                    textAlign: 'center',
                    padding: '24px',
                    background: 'var(--bg-secondary)',
                    borderRadius: '14px',
                    color: 'var(--text-muted)'
                }}>
                    <div style={{ fontSize: '32px', marginBottom: '8px' }}>🍽️</div>
                    <div style={{ fontSize: '14px' }}>No food logged</div>
                </div>
            )}
        </motion.div>
    );

    return (
        <div style={{ background: 'var(--bg-primary)', minHeight: '100vh', paddingBottom: '100px' }}>
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                style={{
                    background: 'var(--bg-gradient-header)',
                    padding: '24px 20px 80px 20px',
                    borderRadius: '0 0 32px 32px'
                }}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <div>
                        <div style={{ color: 'rgba(255,255,255,0.9)', fontSize: '14px', marginBottom: '4px' }}>
                            {greeting.emoji} {greeting.text}
                        </div>
                        <h1 style={{ color: 'white', fontSize: '24px', fontWeight: '700' }}>
                            {user.name || t('onboarding_name')}!
                        </h1>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '12px' }}>
                    {[
                        { icon: Target, label: t('dashboard_goal'), value: `${user.goalCalories} cal` },
                        { icon: Flame, label: 'Burned', value: '0 cal' },
                        { icon: Zap, label: 'Streak', value: '7 days' }
                    ].map((stat, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 }}
                            style={{
                                flex: 1,
                                background: 'rgba(255,255,255,0.15)',
                                backdropFilter: 'blur(10px)',
                                borderRadius: '16px',
                                padding: '14px 12px',
                                textAlign: 'center'
                            }}
                        >
                            <stat.icon size={20} color="white" style={{ marginBottom: '6px' }} />
                            <div style={{ color: 'white', fontWeight: '700', fontSize: '14px' }}>{stat.value}</div>
                            <div style={{ color: 'rgba(255,255,255,0.8)', fontSize: '11px' }}>{stat.label}</div>
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            {/* Calorie Ring Card */}
            <motion.div
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                style={{
                    margin: '-50px 20px 20px 20px',
                    background: 'var(--bg-card)',
                    backdropFilter: 'blur(20px) saturate(200%)',
                    WebkitBackdropFilter: 'blur(12px)',
                    borderRadius: '24px',
                    padding: '24px',
                    boxShadow: 'var(--card-shadow)',
                    border: '1px solid var(--border-color)'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around' }}>
                    <CalorieRing current={totalCalories} goal={user.goalCalories} size={160} />
                    <div style={{ textAlign: 'left' }}>
                        <div style={{ marginBottom: '16px' }}>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>{t('dashboard_remaining')}</div>
                            <div style={{
                                fontSize: '28px',
                                fontWeight: '800',
                                color: remainingCalories < 0 ? '#EF4444' : '#1DB954'
                            }}>
                                {remainingCalories} cal
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: '12px' }}>
                            <div>
                                <div style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>{Math.round(totalProtein)}g</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Protein</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>{Math.round(totalCarbs)}g</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Carbs</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>{Math.round(totalFat)}g</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Fat</div>
                            </div>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Meals Section */}
            <div style={{ padding: '0 20px' }}>
                <MealSection titleKey="dashboard_breakfast" items={meals.breakfast} type="breakfast" />
                <MealSection titleKey="dashboard_lunch" items={meals.lunch} type="lunch" />
                <MealSection titleKey="dashboard_snack" items={meals.snack} type="snack" />
                <MealSection titleKey="dashboard_dinner" items={meals.dinner} type="dinner" />
            </div>

            {/* AI Analysis Button */}
            <div style={{ padding: '0 20px', marginTop: '24px' }}>
                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    style={{
                        width: '100%',
                        padding: '18px',
                        background: 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '16px',
                        fontWeight: '700',
                        fontSize: '16px',
                        cursor: isAnalyzing ? 'wait' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '10px',
                        boxShadow: '0 8px 24px rgba(139, 92, 246, 0.35)'
                    }}
                >
                    {isAnalyzing ? (
                        <span>Processing...</span>
                    ) : (
                        <>✨ Analyze My Day with AI</>
                    )}
                </motion.button>
                {analysis && (
                    <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{
                        marginTop: '16px',
                        padding: '16px',
                        background: 'var(--bg-card)',
                        backdropFilter: 'blur(20px) saturate(200%)',
                        borderRadius: '16px',
                        border: '1px solid var(--border-color)',
                        color: 'var(--text-primary)'
                    }}>
                        {analysis.text}
                    </motion.p>
                )}
            </div>

            {/* Add Food Modal */}
            <AnimatePresence>
                {isAddFoodOpen && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{
                            position: 'fixed',
                            inset: 0,
                            background: 'rgba(0,0,0,0.6)',
                            backdropFilter: 'blur(4px)',
                            zIndex: 10000,
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'flex-end'
                        }}
                    >
                        <motion.div
                            initial={{ y: '100%' }}
                            animate={{ y: 0 }}
                            exit={{ y: '100%' }}
                            transition={{ type: "spring", damping: 25, stiffness: 200 }}
                            style={{
                                background: 'var(--bg-card)',
                                backdropFilter: 'blur(20px) saturate(200%)',
                                borderRadius: '24px 24px 0 0',
                                padding: '24px',
                                height: '85vh',
                                display: 'flex',
                                flexDirection: 'column'
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                                <h3 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    🔍 Add to {selectedMealType?.charAt(0).toUpperCase() + selectedMealType?.slice(1)}
                                </h3>
                                <button onClick={() => { setIsAddFoodOpen(false); setSelectedItemForQuant(null); }}
                                    style={{ background: 'var(--bg-secondary)', border: 'none', borderRadius: '50%', width: '36px', height: '36px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <X size={20} color="var(--text-primary)" />
                                </button>
                            </div>

                            {selectedItemForQuant ? (
                                <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                                    <div style={{ background: 'var(--bg-secondary)', borderRadius: '16px', padding: '24px', textAlign: 'center', marginBottom: '24px' }}>
                                        <div style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '8px' }}>
                                            {selectedItemForQuant["Dish Name"]}
                                        </div>
                                        <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                                            {selectedItemForQuant["Calories (kcal)"]} cal per serving
                                        </div>
                                    </div>

                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '32px', marginBottom: '32px' }}>
                                        <motion.button whileTap={{ scale: 0.9 }} onClick={() => setItemQuantity(q => Math.max(0.5, q - 0.5))}
                                            style={{ width: '56px', height: '56px', borderRadius: '50%', border: 'none', background: 'var(--bg-secondary)', fontSize: '24px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-primary)' }}>
                                            <Minus size={24} />
                                        </motion.button>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '48px', fontWeight: '800', color: '#1DB954' }}>{itemQuantity}</div>
                                            <div style={{ fontSize: '14px', color: 'var(--text-muted)' }}>servings</div>
                                        </div>
                                        <motion.button whileTap={{ scale: 0.9 }} onClick={() => setItemQuantity(q => q + 0.5)}
                                            style={{ width: '56px', height: '56px', borderRadius: '50%', border: 'none', background: '#1DB954', color: 'white', fontSize: '24px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <Plus size={24} />
                                        </motion.button>
                                    </div>

                                    <div style={{ display: 'flex', gap: '12px', marginTop: '20px' }}>
                                        <button onClick={() => setSelectedItemForQuant(null)} style={{ flex: 1, padding: '16px', borderRadius: '16px', border: 'none', background: 'var(--bg-secondary)', fontWeight: '600', cursor: 'pointer', color: 'var(--text-primary)' }}>Back</button>
                                        <motion.button whileTap={{ scale: 0.98 }} onClick={confirmAddFood} style={{ flex: 2, padding: '16px', borderRadius: '16px', border: 'none', background: '#1DB954', color: 'white', fontWeight: '700', cursor: 'pointer' }}>Add Food</motion.button>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    <div style={{ display: 'flex', alignItems: 'center', background: 'var(--input-bg)', borderRadius: '14px', padding: '14px', marginBottom: '20px', border: '1px solid var(--border-color)' }}>
                                        <Search size={20} color="var(--text-muted)" style={{ marginRight: '10px' }} />
                                        <input type="text" placeholder="Search for food..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)}
                                            style={{ border: 'none', background: 'transparent', outline: 'none', flex: 1, fontSize: '16px', color: 'var(--text-primary)' }} autoFocus />
                                    </div>
                                    <div style={{ flex: 1, overflowY: 'auto' }}>
                                        {/* Quick Add Sections - Only show when not searching */}
                                        {searchTerm.length < 3 && (
                                            <>
                                                {/* Popular Foods */}
                                                <div style={{ marginBottom: '24px' }}>
                                                    <h4 style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                        🔥 Popular Foods
                                                    </h4>
                                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                                        {[
                                                            { name: 'Roti', emoji: '🫓' },
                                                            { name: 'Rice', emoji: '🍚' },
                                                            { name: 'Dal', emoji: '🍲' },
                                                            { name: 'Idli', emoji: '🥟' },
                                                            { name: 'Dosa', emoji: '🥞' },
                                                            { name: 'Paratha', emoji: '🫓' },
                                                            { name: 'Poha', emoji: '🍚' },
                                                            { name: 'Upma', emoji: '🍲' },
                                                        ].map((food) => (
                                                            <motion.button
                                                                key={food.name}
                                                                whileTap={{ scale: 0.95 }}
                                                                onClick={() => setSearchTerm(food.name)}
                                                                style={{
                                                                    display: 'flex',
                                                                    alignItems: 'center',
                                                                    gap: '6px',
                                                                    padding: '10px 14px',
                                                                    borderRadius: '20px',
                                                                    border: '1px solid var(--border-color)',
                                                                    background: 'var(--bg-secondary)',
                                                                    cursor: 'pointer',
                                                                    fontSize: '13px',
                                                                    fontWeight: '500',
                                                                    color: 'var(--text-primary)'
                                                                }}
                                                            >
                                                                <span>{food.emoji}</span>
                                                                <span style={{ fontWeight: '600' }}>{food.name}</span>
                                                            </motion.button>
                                                        ))}
                                                    </div>
                                                </div>

                                                {/* Healthy Choices */}
                                                <div style={{ marginBottom: '24px' }}>
                                                    <h4 style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                        🥗 Healthy Choices
                                                    </h4>
                                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                                        {[
                                                            { name: 'Sprouts', emoji: '🥗' },
                                                            { name: 'Oats', emoji: '🥣' },
                                                            { name: 'Salad', emoji: '🥬' },
                                                            { name: 'Fruit', emoji: '🍎' },
                                                            { name: 'Moong Dal', emoji: '🫘' },
                                                        ].map((food) => (
                                                            <motion.button
                                                                key={food.name}
                                                                whileTap={{ scale: 0.95 }}
                                                                onClick={() => setSearchTerm(food.name)}
                                                                style={{
                                                                    display: 'flex',
                                                                    alignItems: 'center',
                                                                    gap: '6px',
                                                                    padding: '10px 14px',
                                                                    borderRadius: '20px',
                                                                    border: '1px solid var(--border-color)',
                                                                    background: 'var(--bg-secondary)',
                                                                    cursor: 'pointer',
                                                                    fontSize: '13px',
                                                                    fontWeight: '500',
                                                                    color: 'var(--text-primary)'
                                                                }}
                                                            >
                                                                <span>{food.emoji}</span>
                                                                <span style={{ fontWeight: '600' }}>{food.name}</span>
                                                            </motion.button>
                                                        ))}
                                                    </div>
                                                </div>

                                                <p style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px', marginBottom: '16px' }}>
                                                    or type to search 3500+ foods
                                                </p>
                                            </>
                                        )}

                                        {/* Search Results */}
                                        {!isSearching && searchResults.map((item, idx) => (
                                            <motion.div key={idx} whileHover={{ background: 'var(--bg-secondary)' }} onClick={() => handleFoodClick(item)}
                                                style={{ padding: '16px', borderBottom: '1px solid var(--border-color)', cursor: 'pointer', borderRadius: '12px' }}>
                                                <div style={{ fontWeight: '600', marginBottom: '4px', color: 'var(--text-primary)' }}>{item["Dish Name"]}</div>
                                                <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{item["Calories (kcal)"]} cal • {item["Serving Size"]}</div>
                                            </motion.div>
                                        ))}
                                    </div>
                                </>
                            )}
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            <BottomNav />
        </div>
    );
}
