"use client";
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Calendar, Star, Plus, Minus, Check, Clock, Utensils, ChevronRight, Coffee, Flame, Zap, Heart } from 'lucide-react';
import { useUser } from '../../context/UserContext';
import { useLanguage } from '../../context/LanguageContext';
import { useTheme } from '../../context/ThemeContext';
import BottomNav from '../../components/BottomNav';
import confetti from 'canvas-confetti';

// Food emoji mapping for visual appeal
const foodEmojis = {
    'Coffee': '☕', 'Chai': '🍵', 'Tea': '🍵',
    'Poha': '🍚', 'Aloo Paratha': '🫓', 'Idli Sambar': '🍛',
    'Bread Omelette': '🍳', 'Puri Bhaji': '🫓', 'Masala Dosa': '🥞',
    'Chole Bhature': '🍛', 'Dal': '🍲', 'Rice': '🍚',
    'Roti': '🫓', 'Achaar': '🥒', 'Paneer': '🧀',
    'Chicken': '🍗', 'Egg': '🥚', 'Fish': '🐟',
    'Mutton': '🍖', 'Biryani': '🍛', 'Samosa': '🥟',
    'Pakora': '🍘', 'Vada Pav': '🍔', 'Pani Puri': '🥣',
    default: '🍽️'
};

const getEmoji = (name) => {
    for (const [key, emoji] of Object.entries(foodEmojis)) {
        if (name.toLowerCase().includes(key.toLowerCase())) return emoji;
    }
    return foodEmojis.default;
};

// Calorie color coding
const getCalorieColor = (cal) => {
    if (cal < 150) return { bg: '#E8F5E9', text: '#2E7D32' };
    if (cal < 300) return { bg: '#FFF3E0', text: '#E65100' };
    return { bg: '#FFEBEE', text: '#C62828' };
};

export default function Mess() {
    const { addFood } = useUser();
    const { t } = useLanguage();
    const { isDark } = useTheme();
    const [selectedMeal, setSelectedMeal] = useState('lunch');
    const [selectedDay, setSelectedDay] = useState(new Date().getDay() === 0 ? 6 : new Date().getDay() - 1);
    const [quantities, setQuantities] = useState({});
    const [bookedMeals, setBookedMeals] = useState({});
    const [isLoaded, setIsLoaded] = useState(false);
    const [currentDate, setCurrentDate] = useState(new Date());

    useEffect(() => {
        const savedBooked = localStorage.getItem('aahar_mess_booked');
        const savedDate = localStorage.getItem('aahar_mess_date');
        const today = new Date().toISOString().split('T')[0];
        if (savedDate === today && savedBooked) {
            setBookedMeals(JSON.parse(savedBooked));
        } else {
            setBookedMeals({});
            localStorage.setItem('aahar_mess_booked', '{}');
            localStorage.setItem('aahar_mess_date', today);
        }
        setIsLoaded(true);
    }, []);

    useEffect(() => {
        if (isLoaded) {
            const today = new Date().toISOString().split('T')[0];
            localStorage.setItem('aahar_mess_booked', JSON.stringify(bookedMeals));
            localStorage.setItem('aahar_mess_date', today);
        }
    }, [bookedMeals, isLoaded]);

    const getTodayIndex = () => {
        const day = currentDate.getDay();
        return day === 0 ? 6 : day - 1;
    };

    const todayIndex = getTodayIndex();
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    const getWeekDates = () => {
        const today = new Date();
        const dates = [];
        for (let i = 0; i < 7; i++) {
            const date = new Date(today);
            date.setDate(today.getDate() - today.getDay() + (i + 1));
            dates.push(date.getDate());
        }
        return dates;
    };
    const dates = getWeekDates();

    const messMenu = {
        breakfast: {
            staples: [
                { name: 'Coffee', unit: 'cup', calories: 80, protein: 2, carbs: 12, fat: 3, desc: 'Hot brewed coffee', unlimited: true },
                { name: 'Chai', unit: 'cup', calories: 60, protein: 2, carbs: 10, fat: 2, desc: 'Masala tea', unlimited: true },
            ],
            weekly: {
                0: { name: 'Poha', calories: 250, protein: 6, carbs: 42, fat: 8, rating: 4.2, desc: 'Flattened rice with peanuts', popular: true },
                1: { name: 'Aloo Paratha', calories: 320, protein: 8, carbs: 50, fat: 12, rating: 4.3, desc: 'Stuffed flatbread' },
                2: { name: 'Idli Sambar', calories: 280, protein: 10, carbs: 46, fat: 6, rating: 4.4, desc: 'Steamed rice cakes', popular: true },
                3: { name: 'Bread Omelette', calories: 260, protein: 14, carbs: 30, fat: 10, rating: 4.3, desc: 'Egg sandwich' },
                4: { name: 'Puri Bhaji', calories: 380, protein: 9, carbs: 58, fat: 14, rating: 4.5, desc: 'Fried bread curry' },
                5: { name: 'Masala Dosa', calories: 360, protein: 9, carbs: 55, fat: 12, rating: 4.7, desc: 'Crispy crepe', popular: true },
                6: { name: 'Chole Bhature', calories: 480, protein: 14, carbs: 68, fat: 16, rating: 4.9, desc: 'Chickpea curry fried bread', popular: true },
            }
        },
        lunch: {
            staples: [
                { name: 'Dal', unit: 'bowl', calories: 180, protein: 12, carbs: 28, fat: 3, desc: 'Yellow dal tadka', unlimited: true },
                { name: 'Rice', unit: '150g', calories: 200, protein: 4, carbs: 44, fat: 0, desc: 'Steamed white rice', unlimited: true },
                { name: 'Roti', unit: 'pc', calories: 70, protein: 2.5, carbs: 14, fat: 1, desc: 'Whole wheat roti', unlimited: true },
                { name: 'Achaar', unit: 'srv', calories: 15, protein: 0, carbs: 3, fat: 0, desc: 'Pickle', unlimited: true },
            ],
            weekly: {
                0: { type: 'veg', items: [{ name: 'Aloo Sabzi', calories: 180, protein: 4, carbs: 34, fat: 6, rating: 4.0, desc: 'Potato curry' }] },
                1: { type: 'choice', veg: { name: 'Paneer Sabzi', calories: 220, protein: 12, carbs: 8, fat: 16, rating: 4.2 }, nonVeg: { name: 'Chicken Curry', calories: 280, protein: 25, carbs: 10, fat: 16, rating: 4.4, popular: true } },
                2: { type: 'choice', veg: { name: 'Mix Veg', calories: 160, protein: 5, carbs: 20, fat: 7, rating: 3.9 }, nonVeg: { name: 'Egg Curry', calories: 200, protein: 14, carbs: 8, fat: 13, rating: 4.1 } },
                3: { type: 'veg', items: [{ name: 'Rajma', calories: 240, protein: 14, carbs: 42, fat: 4, rating: 4.3, desc: 'Kidney beans', popular: true }] },
                4: { type: 'choice', veg: { name: 'Aloo Gobi', calories: 170, protein: 4, carbs: 25, fat: 7, rating: 4.0 }, nonVeg: { name: 'Fish Curry', calories: 260, protein: 22, carbs: 8, fat: 16, rating: 4.3 } },
                5: { type: 'choice', veg: { name: 'Chole', calories: 240, protein: 12, carbs: 36, fat: 6, rating: 4.5, popular: true }, nonVeg: { name: 'Chicken Masala', calories: 300, protein: 26, carbs: 12, fat: 18, rating: 4.6 } },
                6: { type: 'choice', veg: { name: 'Paneer Butter Masala', calories: 280, protein: 14, carbs: 12, fat: 20, rating: 4.7, popular: true }, nonVeg: { name: 'Mutton Curry', calories: 340, protein: 28, carbs: 10, fat: 22, rating: 4.8 } },
            }
        },
        snack: {
            staples: [
                { name: 'Coffee', unit: 'cup', calories: 80, protein: 2, carbs: 12, fat: 3, desc: 'Hot coffee', unlimited: true },
                { name: 'Tea', unit: 'cup', calories: 60, protein: 2, carbs: 10, fat: 2, desc: 'Milk tea', unlimited: true },
            ],
            weekly: {
                0: { name: 'Samosa', calories: 240, rating: 4.3, protein: 5, carbs: 28, fat: 12 },
                1: { name: 'Pakora', calories: 210, rating: 4.2, protein: 4, carbs: 22, fat: 12 },
                2: { name: 'Bread Pakoda', calories: 220, rating: 4.1, protein: 5, carbs: 24, fat: 11 },
                3: { name: 'Kachori', calories: 260, rating: 4.3, protein: 5, carbs: 30, fat: 14 },
                4: { name: 'Cutlet', calories: 200, rating: 4.2, protein: 6, carbs: 20, fat: 10 },
                5: { name: 'Vada Pav', calories: 290, rating: 4.6, protein: 6, carbs: 36, fat: 14, popular: true },
                6: { name: 'Pani Puri', calories: 180, rating: 4.8, protein: 3, carbs: 28, fat: 6, popular: true },
            }
        },
        dinner: {
            staples: [
                { name: 'Dal', unit: 'bowl', calories: 180, protein: 12, carbs: 28, fat: 3, desc: 'Yellow dal tadka', unlimited: true },
                { name: 'Rice', unit: '150g', calories: 200, protein: 4, carbs: 44, fat: 0, desc: 'Steamed white rice', unlimited: true },
                { name: 'Roti', unit: 'pc', calories: 70, protein: 2.5, carbs: 14, fat: 1, desc: 'Whole wheat roti', unlimited: true },
            ],
            weekly: {
                0: { type: 'veg', items: [{ name: 'Seasonal Sabzi', calories: 150, protein: 4, carbs: 20, fat: 6, rating: 3.9 }] },
                1: { type: 'choice', veg: { name: 'Paneer Do Pyaza', calories: 240, protein: 12, carbs: 10, fat: 18, rating: 4.1 }, nonVeg: { name: 'Chicken Curry', calories: 280, protein: 25, carbs: 10, fat: 16, rating: 4.3 } },
                2: { type: 'veg', items: [{ name: 'Aloo Matar', calories: 180, protein: 5, carbs: 28, fat: 6, rating: 4.0 }] },
                3: { type: 'veg', items: [{ name: 'Chana Masala', calories: 220, protein: 10, carbs: 34, fat: 5, rating: 4.2 }] },
                4: { type: 'special', veg: { name: 'Veg Biryani', calories: 420, protein: 10, carbs: 68, fat: 12, rating: 4.7 }, nonVeg: { name: 'Chicken Biryani', calories: 520, protein: 24, carbs: 64, fat: 20, rating: 4.9, popular: true }, extras: [{ name: 'Raita', calories: 60, protein: 3, carbs: 4, fat: 3 }] },
                5: { type: 'choice', veg: { name: 'Palak Paneer', calories: 240, protein: 14, carbs: 10, fat: 16, rating: 4.4, popular: true }, nonVeg: { name: 'Egg Bhurji', calories: 220, protein: 16, carbs: 6, fat: 16, rating: 4.2 } },
                6: { type: 'veg', items: [{ name: 'Dal Makhani', calories: 260, protein: 12, carbs: 32, fat: 10, rating: 4.6, popular: true }] },
            }
        }
    };

    const mealIcons = { breakfast: '🌅', lunch: '☀️', snack: '🍪', dinner: '🌙' };

    const updateQuantity = (itemName, delta) => {
        const key = `${selectedMeal}-${itemName}`;
        setQuantities(prev => {
            const currentQty = prev[key] || 0;
            const newQty = Math.max(0, currentQty + delta);
            return { ...prev, [key]: newQty };
        });
    };

    const toggleMainItem = (itemName) => {
        if (bookedMeals[selectedMeal] && selectedDay === todayIndex) return;
        const menu = messMenu[selectedMeal];
        const special = menu.weekly[selectedDay];

        if (special && (special.type === 'choice' || special.type === 'special')) {
            const vegName = special.veg?.name;
            const nonVegName = special.nonVeg?.name;
            const vegKey = `${selectedMeal}-${vegName}`;
            const nonVegKey = `${selectedMeal}-${nonVegName}`;
            const itemKey = `${selectedMeal}-${itemName}`;

            setQuantities(prev => {
                const newState = { ...prev };
                if (!newState[itemKey]) {
                    newState[itemKey] = 1;
                    if (itemName === vegName && nonVegName) newState[nonVegKey] = 0;
                    if (itemName === nonVegName && vegName) newState[vegKey] = 0;
                } else {
                    newState[itemKey] = 0;
                }
                return newState;
            });
        } else {
            const key = `${selectedMeal}-${itemName}`;
            setQuantities(prev => ({ ...prev, [key]: prev[key] ? 0 : 1 }));
        }
    };

    // Calculate meal summary
    const getMealSummary = () => {
        let totalCal = 0, totalProtein = 0, totalCarbs = 0, totalFat = 0, itemCount = 0;
        const menu = messMenu[selectedMeal];
        const special = menu.weekly[selectedDay];

        menu.staples.forEach(staple => {
            const key = `${selectedMeal}-${staple.name}`;
            const qty = quantities[key] || 0;
            if (qty > 0) {
                totalCal += staple.calories * qty;
                totalProtein += (staple.protein || 0) * qty;
                totalCarbs += (staple.carbs || 0) * qty;
                totalFat += (staple.fat || 0) * qty;
                itemCount += qty;
            }
        });

        if (special) {
            const checkItem = (item) => {
                if (!item) return;
                const key = `${selectedMeal}-${item.name}`;
                if (quantities[key]) {
                    totalCal += item.calories || 0;
                    totalProtein += item.protein || 0;
                    totalCarbs += item.carbs || 0;
                    totalFat += item.fat || 0;
                    itemCount++;
                }
            };
            if (special.veg) checkItem(special.veg);
            if (special.nonVeg) checkItem(special.nonVeg);
            if (special.items) special.items.forEach(checkItem);
            if (special.name) checkItem(special);
        }

        return { totalCal, totalProtein, totalCarbs, totalFat, itemCount };
    };

    const handleConfirmMeal = () => {
        const itemsToLog = [];
        const menu = messMenu[selectedMeal];
        const special = menu.weekly[selectedDay];

        menu.staples.forEach(staple => {
            const key = `${selectedMeal}-${staple.name}`;
            const qty = quantities[key] || 0;
            if (qty > 0) itemsToLog.push({ ...staple, quantity: qty });
        });

        if (special) {
            if (special.type === 'choice' || special.type === 'special') {
                const vegKey = `${selectedMeal}-${special.veg?.name}`;
                const nonVegKey = `${selectedMeal}-${special.nonVeg?.name}`;
                if (quantities[vegKey]) itemsToLog.push({ ...special.veg, quantity: 1 });
                if (quantities[nonVegKey]) itemsToLog.push({ ...special.nonVeg, quantity: 1 });
                if (special.extras) {
                    special.extras.forEach(extra => {
                        if (quantities[vegKey] || quantities[nonVegKey]) itemsToLog.push({ ...extra, quantity: 1 });
                    });
                }
            } else if (special.name) {
                const key = `${selectedMeal}-${special.name}`;
                if (quantities[key]) itemsToLog.push({ ...special, quantity: 1 });
            } else if (special.items) {
                special.items.forEach(item => {
                    const key = `${selectedMeal}-${item.name}`;
                    if (quantities[key]) itemsToLog.push({ ...item, quantity: 1 });
                });
            }
        }

        itemsToLog.forEach(item => {
            addFood(selectedMeal, {
                name: item.name,
                calories: item.calories * item.quantity,
                protein: (item.protein || 0) * item.quantity,
                carbs: (item.carbs || 0) * item.quantity,
                fat: (item.fat || 0) * item.quantity
            });
        });

        setBookedMeals({ ...bookedMeals, [selectedMeal]: true });

        confetti({ particleCount: 50, spread: 70, origin: { y: 0.6 }, colors: ['#1DB954', '#16a34a', '#22c55e', '#4ade80'] });

        const newQuantities = { ...quantities };
        Object.keys(newQuantities).forEach(key => {
            if (key.startsWith(selectedMeal)) delete newQuantities[key];
        });
        setQuantities(newQuantities);
    };

    const renderStapleCard = (item, index) => {
        const key = `${selectedMeal}-${item.name}`;
        const qty = quantities[key] || 0;
        const isToday = selectedDay === todayIndex;
        const isBooked = bookedMeals[selectedMeal] && isToday;
        const calColor = getCalorieColor(item.calories);

        return (
            <motion.div
                key={item.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                style={{
                    background: 'var(--bg-card)',
                    backdropFilter: 'blur(20px) saturate(200%)',
                    borderRadius: '16px',
                    padding: '16px',
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    border: qty > 0 ? '2px solid #1DB954' : '1px solid var(--border-color)',
                    boxShadow: qty > 0 ? '0 4px 12px rgba(29, 185, 84, 0.15)' : 'var(--shadow-sm)',
                    transition: 'all 0.3s ease'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                        width: '48px', height: '48px', borderRadius: '12px',
                        background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '24px'
                    }}>
                        {getEmoji(item.name)}
                    </div>
                    <div>
                        <div style={{ fontWeight: '600', color: 'var(--text-primary)', fontSize: '15px', marginBottom: '4px' }}>
                            {item.name}
                            {item.unlimited && <span style={{ fontSize: '10px', color: '#1DB954', marginLeft: '6px', background: 'rgba(29, 185, 84, 0.1)', padding: '2px 6px', borderRadius: '4px' }}>∞</span>}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '12px', padding: '3px 8px', borderRadius: '6px', background: calColor.bg, color: calColor.text, fontWeight: '600' }}>
                                {item.calories} Cal
                            </span>
                            <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>per {item.unit}</span>
                        </div>
                    </div>
                </div>

                {isToday && !isBooked && (
                    <motion.div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-secondary)', borderRadius: '12px', padding: '6px' }}>
                        <motion.button
                            whileTap={{ scale: 0.9 }}
                            onClick={() => updateQuantity(item.name, -1)}
                            disabled={qty === 0}
                            style={{
                                width: '32px', height: '32px', borderRadius: '8px', border: 'none',
                                background: qty > 0 ? 'var(--bg-primary)' : 'transparent', color: qty > 0 ? 'var(--text-primary)' : 'var(--text-muted)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                cursor: qty === 0 ? 'default' : 'pointer', boxShadow: qty > 0 ? 'var(--shadow-sm)' : 'none'
                            }}
                        >
                            <Minus size={16} />
                        </motion.button>
                        <motion.span
                            key={qty}
                            initial={{ scale: 1.3 }}
                            animate={{ scale: 1 }}
                            style={{ fontWeight: '700', minWidth: '24px', textAlign: 'center', fontSize: '16px', color: 'var(--text-primary)' }}
                        >
                            {qty}
                        </motion.span>
                        <motion.button
                            whileTap={{ scale: 0.9 }}
                            onClick={() => updateQuantity(item.name, 1)}
                            style={{
                                width: '32px', height: '32px', borderRadius: '8px', border: 'none',
                                background: '#1DB954', color: 'white',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                cursor: 'pointer', boxShadow: '0 2px 4px rgba(29,185,84,0.3)'
                            }}
                        >
                            <Plus size={16} />
                        </motion.button>
                    </motion.div>
                )}
            </motion.div>
        );
    };

    const renderMainCard = (item, index) => {
        if (!item) return null;
        const key = `${selectedMeal}-${item.name}`;
        const isSelected = quantities[key] > 0;
        const isToday = selectedDay === todayIndex;
        const isBooked = bookedMeals[selectedMeal] && isToday;
        const calColor = getCalorieColor(item.calories);

        return (
            <motion.div
                key={item.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + index * 0.05 }}
                whileHover={{ scale: isToday && !isBooked ? 1.01 : 1 }}
                whileTap={{ scale: isToday && !isBooked ? 0.99 : 1 }}
                onClick={() => isToday && !isBooked && toggleMainItem(item.name)}
                style={{
                    background: isSelected ? 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)' : 'var(--bg-card)',
                    backdropFilter: 'blur(20px) saturate(200%)',
                    borderRadius: '16px',
                    padding: '16px',
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    border: isSelected ? '2px solid #1DB954' : '1px solid var(--border-color)',
                    cursor: isToday && !isBooked ? 'pointer' : 'default',
                    boxShadow: isSelected ? '0 4px 12px rgba(29, 185, 84, 0.15)' : 'var(--shadow-sm)',
                    position: 'relative',
                    overflow: 'hidden'
                }}
            >
                {item.popular && (
                    <div style={{
                        position: 'absolute', top: '8px', right: '8px',
                        background: 'linear-gradient(135deg, #FF6B6B, #FF8E53)',
                        color: 'white', fontSize: '9px', fontWeight: '700',
                        padding: '3px 8px', borderRadius: '6px',
                        display: 'flex', alignItems: 'center', gap: '3px'
                    }}>
                        <Flame size={10} /> HOT
                    </div>
                )}

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                    {isToday && (
                        <motion.div
                            animate={{ scale: isSelected ? 1 : 1, borderWidth: isSelected ? '6px' : '2px' }}
                            style={{
                                width: '24px', height: '24px', borderRadius: '50%',
                                border: `solid ${isSelected ? '#1DB954' : 'var(--border-color)'}`,
                                background: isSelected ? '#1DB954' : 'transparent',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                flexShrink: 0
                            }}
                        >
                            {isSelected && <Check size={12} color="white" />}
                        </motion.div>
                    )}
                    <div style={{
                        width: '48px', height: '48px', borderRadius: '12px',
                        background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '24px', flexShrink: 0
                    }}>
                        {getEmoji(item.name)}
                    </div>
                    <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: '600', color: isSelected ? '#1a1a1a' : 'var(--text-primary)', fontSize: '15px', marginBottom: '4px' }}>
                            {item.name}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '12px', padding: '3px 8px', borderRadius: '6px', background: calColor.bg, color: calColor.text, fontWeight: '600' }}>
                                {item.calories} Cal
                            </span>
                            {item.rating && (
                                <span style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '12px', color: '#f59e0b', fontWeight: '600' }}>
                                    <Star size={12} fill="#f59e0b" /> {item.rating}
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            </motion.div>
        );
    };

    const currentMenu = messMenu[selectedMeal];
    const dailySpecial = currentMenu.weekly[selectedDay];
    const isToday = selectedDay === todayIndex;
    const isBooked = bookedMeals[selectedMeal] && isToday;
    const hasSelection = isToday && Object.keys(quantities).filter(k => k.startsWith(selectedMeal) && quantities[k] > 0).length > 0;
    const summary = getMealSummary();

    return (
        <>
            <div style={{ background: 'var(--bg-primary)', minHeight: '100vh', paddingBottom: '180px' }}>
                <div style={{ maxWidth: '600px', margin: '0 auto', padding: '20px' }}>

                    {/* Header */}
                    <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: '20px' }}>
                        <h1 style={{ fontSize: '26px', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                            {mealIcons[selectedMeal]} {t('mess_menu') || 'Mess Menu'}
                        </h1>
                        <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                            {isToday ? "Select items for today's meal" : "Viewing menu for another day"}
                        </p>
                    </motion.div>

                    {/* Day Selector */}
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                        style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '16px', marginBottom: '16px' }}>
                        {days.map((day, index) => {
                            const isSelected = selectedDay === index;
                            const isTodayDay = index === todayIndex;
                            return (
                                <motion.button
                                    key={day}
                                    whileTap={{ scale: 0.95 }}
                                    onClick={() => setSelectedDay(index)}
                                    style={{
                                        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                                        padding: '10px 14px', borderRadius: '14px',
                                        border: isSelected ? 'none' : '1px solid var(--border-color)',
                                        background: isSelected ? 'linear-gradient(135deg, #1DB954, #16a34a)' : 'var(--bg-card)',
                                        backdropFilter: isSelected ? 'none' : 'blur(12px)',
                                        color: isSelected ? 'white' : 'var(--text-secondary)',
                                        flexShrink: 0, minWidth: '54px', position: 'relative', cursor: 'pointer',
                                        boxShadow: isSelected ? '0 4px 12px rgba(29,185,84,0.3)' : 'var(--shadow-sm)'
                                    }}
                                >
                                    <span style={{ fontSize: '11px', fontWeight: '500', marginBottom: '2px' }}>{day}</span>
                                    <span style={{ fontSize: '18px', fontWeight: '700', color: isSelected ? 'white' : 'var(--text-primary)' }}>{dates[index]}</span>
                                    {isTodayDay && <span style={{ position: 'absolute', bottom: '5px', width: '5px', height: '5px', background: isSelected ? 'white' : '#1DB954', borderRadius: '50%' }} />}
                                </motion.button>
                            );
                        })}
                    </motion.div>

                    {/* Meal Type Selector */}
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
                        style={{
                            background: 'var(--bg-card)',
                            backdropFilter: 'blur(20px) saturate(200%)',
                            padding: '5px',
                            borderRadius: '16px',
                            display: 'flex',
                            marginBottom: '24px',
                            boxShadow: 'var(--shadow-sm)',
                            position: 'relative'
                        }}>
                        <motion.div
                            layoutId="mealIndicator"
                            style={{
                                position: 'absolute', top: '5px',
                                left: `calc(${['breakfast', 'lunch', 'snack', 'dinner'].indexOf(selectedMeal) * 25}% + 5px)`,
                                width: 'calc(25% - 10px)', height: 'calc(100% - 10px)',
                                background: 'linear-gradient(135deg, #1DB954, #16a34a)',
                                borderRadius: '12px', zIndex: 0
                            }}
                            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                        />
                        {['breakfast', 'lunch', 'snack', 'dinner'].map((meal) => (
                            <button
                                key={meal}
                                onClick={() => setSelectedMeal(meal)}
                                style={{
                                    flex: 1, padding: '12px 8px', borderRadius: '12px', border: 'none',
                                    background: 'transparent', color: selectedMeal === meal ? 'white' : 'var(--text-secondary)',
                                    fontWeight: '600', fontSize: '13px', cursor: 'pointer',
                                    position: 'relative', zIndex: 1, transition: 'color 0.2s'
                                }}
                            >
                                <span style={{ marginRight: '4px' }}>{mealIcons[meal]}</span>
                                {t(meal) || meal.charAt(0).toUpperCase() + meal.slice(1)}
                            </button>
                        ))}
                    </motion.div>

                    {/* Menu Items */}
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                        <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Utensils size={18} /> {t('select_items') || 'Select Items'}
                        </h2>

                        {/* Staples */}
                        <div style={{ marginBottom: '20px' }}>
                            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '10px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                Unlimited Items
                            </p>
                            {currentMenu.staples.map((item, i) => renderStapleCard(item, i))}
                        </div>

                        {/* Specials */}
                        {dailySpecial && (
                            <div>
                                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '10px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                    Today's Special {dailySpecial.type === 'choice' && '(Pick One)'}
                                </p>
                                {(dailySpecial.type === 'choice' || dailySpecial.type === 'special') ? (
                                    <>
                                        {renderMainCard(dailySpecial.veg, 0)}
                                        {dailySpecial.nonVeg && renderMainCard(dailySpecial.nonVeg, 1)}
                                    </>
                                ) : dailySpecial.items ? (
                                    dailySpecial.items.map((item, i) => renderMainCard(item, i))
                                ) : (
                                    renderMainCard(dailySpecial, 0)
                                )}
                            </div>
                        )}
                    </motion.div>
                </div>

                {/* Floating Meal Summary */}
                <AnimatePresence>
                    {(hasSelection || isBooked) && (
                        <motion.div
                            initial={{ y: 100, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            exit={{ y: 100, opacity: 0 }}
                            style={{
                                position: 'fixed', bottom: '90px', left: '20px', right: '20px',
                                maxWidth: '560px',
                                margin: '0 auto',
                                background: 'var(--bg-card)',
                                backdropFilter: 'blur(12px)',
                                borderRadius: '20px',
                                padding: '16px 20px', boxShadow: '0 -4px 30px rgba(0,0,0,0.15)',
                                zIndex: 100, border: '1px solid var(--border-color)'
                            }}
                        >
                            {!isBooked && (
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                                    <div style={{ display: 'flex', gap: '16px' }}>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1DB954' }}>{summary.totalCal}</div>
                                            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Cal</div>
                                        </div>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '18px', fontWeight: '700', color: '#3b82f6' }}>{summary.totalProtein}g</div>
                                            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Protein</div>
                                        </div>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '18px', fontWeight: '700', color: '#f59e0b' }}>{summary.totalCarbs}g</div>
                                            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Carbs</div>
                                        </div>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{summary.itemCount} items</div>
                                    </div>
                                </div>
                            )}
                            <motion.button
                                whileTap={{ scale: 0.98 }}
                                onClick={handleConfirmMeal}
                                disabled={!hasSelection && isBooked}
                                style={{
                                    width: '100%', padding: '14px',
                                    background: isBooked ? '#E8F5E9' : 'linear-gradient(135deg, #1DB954, #16a34a)',
                                    color: isBooked ? '#1DB954' : 'white',
                                    border: 'none', borderRadius: '14px',
                                    fontWeight: '700', fontSize: '15px',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                                    cursor: hasSelection ? 'pointer' : 'default',
                                    boxShadow: isBooked ? 'none' : '0 4px 15px rgba(29,185,84,0.4)'
                                }}
                            >
                                {isBooked && !hasSelection ? <><Check size={20} /> Meal Logged!</> : <><Utensils size={20} /> Log {selectedMeal.charAt(0).toUpperCase() + selectedMeal.slice(1)}</>}
                            </motion.button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
            <BottomNav />
        </>
    );
}
