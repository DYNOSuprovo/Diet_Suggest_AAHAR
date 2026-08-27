"use client";
import { createContext, useContext, useState, useEffect } from 'react';

const UserContext = createContext();

export function UserProvider({ children }) {
    // Default empty state
    const defaultUser = {
        name: '',
        email: '',
        weight: 0,
        height: 0,
        age: 0,
        gender: 'male',
        activityLevel: 'sedentary',
        bmi: 0,
        goalCalories: 0,
        goalWater: 0,
    };

    const [user, setUser] = useState(defaultUser);
    const [isOnboarded, setIsOnboarded] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    const [meals, setMeals] = useState({
        breakfast: [],
        lunch: [],
        snack: [],
        dinner: []
    });

    const [water, setWater] = useState({
        current: 0,
        goal: 2000,
        history: []
    });

    const [preferences, setPreferences] = useState({
        vegetarian: false,
        glutenFree: false,
        dairyFree: false,
        lowCarb: false,
    });

    // Daily Stats History
    const [dailyStats, setDailyStats] = useState({});
    const [lastActiveDate, setLastActiveDate] = useState('');

    // Load from LocalStorage on mount
    useEffect(() => {
        const loadData = () => {
            try {
                const savedUser = localStorage.getItem('aahar_user');
                const savedOnboarded = localStorage.getItem('aahar_onboarded');
                const savedMeals = localStorage.getItem('aahar_meals');
                const savedWater = localStorage.getItem('aahar_water');
                const savedPreferences = localStorage.getItem('aahar_preferences');
                const savedDailyStats = localStorage.getItem('aahar_daily_stats');
                const savedLastDate = localStorage.getItem('aahar_last_date');

                if (savedUser) setUser(JSON.parse(savedUser));
                if (savedOnboarded) setIsOnboarded(JSON.parse(savedOnboarded));
                if (savedMeals) setMeals(JSON.parse(savedMeals));
                if (savedWater) setWater(JSON.parse(savedWater));
                if (savedPreferences) setPreferences(JSON.parse(savedPreferences));
                if (savedDailyStats) setDailyStats(JSON.parse(savedDailyStats));

                // Check for new day
                const today = new Date().toISOString().split('T')[0];
                if (savedLastDate && savedLastDate !== today) {
                    // Reset daily trackers for new day
                    setWater({ current: 0, goal: 2000, history: [] });
                    setMeals({ breakfast: [], lunch: [], snack: [], dinner: [] });
                }
                setLastActiveDate(today);

            } catch (error) {
                console.error("Failed to load data", error);
            } finally {
                setIsLoading(false);
            }
        };
        loadData();
    }, []);

    // Save to LocalStorage whenever state changes
    useEffect(() => {
        if (!isLoading) {
            localStorage.setItem('aahar_user', JSON.stringify(user));
            localStorage.setItem('aahar_onboarded', JSON.stringify(isOnboarded));
            localStorage.setItem('aahar_meals', JSON.stringify(meals));
            localStorage.setItem('aahar_water', JSON.stringify(water));
            localStorage.setItem('aahar_preferences', JSON.stringify(preferences));
            localStorage.setItem('aahar_daily_stats', JSON.stringify(dailyStats));
            localStorage.setItem('aahar_last_date', lastActiveDate);
        }
    }, [user, isOnboarded, meals, water, preferences, dailyStats, lastActiveDate, isLoading]);

    // Sync current state to dailyStats whenever it changes
    useEffect(() => {
        if (!lastActiveDate || isLoading) return;

        setDailyStats(prev => ({
            ...prev,
            [lastActiveDate]: {
                weight: user.weight,
                water: water.current
            }
        }));
    }, [water.current, user.weight, lastActiveDate, isLoading]);

    // Automatic Date Check (Periodically & On Focus)
    useEffect(() => {
        if (isLoading || !lastActiveDate) return;

        const checkDate = () => {
            const today = new Date().toISOString().split('T')[0];
            if (today !== lastActiveDate) {
                console.log("New day detected! Resetting...");
                setLastActiveDate(today);
                setWater({ current: 0, goal: water.goal || 2000, history: [] });
                setMeals({ breakfast: [], lunch: [], snack: [], dinner: [] });
            }
        };

        // Check every minute
        const interval = setInterval(checkDate, 60000);

        // Check when app comes to foreground
        const handleVisibilityChange = () => {
            if (document.visibilityState === 'visible') {
                checkDate();
            }
        };
        document.addEventListener('visibilitychange', handleVisibilityChange);

        return () => {
            clearInterval(interval);
            document.removeEventListener('visibilitychange', handleVisibilityChange);
        };
    }, [lastActiveDate, isLoading, water.goal]);


    const addWater = (amount) => {
        const today = new Date().toISOString().split('T')[0];
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        if (today !== lastActiveDate) {
            setLastActiveDate(today);
            setMeals({ breakfast: [], lunch: [], snack: [], dinner: [] });
            setWater({
                current: amount,
                goal: water.goal || 2000,
                history: [{ amount, time }]
            });
        } else {
            setWater(prev => {
                const effectiveGoal = user.goalWater > 0 ? user.goalWater : 2000;
                const maxWater = effectiveGoal * 1.5;
                const currentTotal = prev.history.reduce((sum, item) => sum + item.amount, 0);
                if (currentTotal >= maxWater) return prev;

                const newHistory = [{ amount, time }, ...prev.history];
                const totalHistory = newHistory.reduce((sum, item) => sum + item.amount, 0);

                return {
                    current: Math.min(totalHistory, maxWater),
                    goal: prev.goal || 2000,
                    history: newHistory
                };
            });
        }
    };

    const removeWater = (index) => {
        setWater(prev => {
            const newHistory = prev.history.filter((_, i) => i !== index);
            const totalHistory = newHistory.reduce((sum, item) => sum + item.amount, 0);
            const effectiveGoal = user.goalWater > 0 ? user.goalWater : 2000;
            const maxWater = effectiveGoal * 1.5;

            return {
                current: Math.min(totalHistory, maxWater),
                goal: prev.goal || 2000,
                history: newHistory
            };
        });
    };

    const addFood = (mealType, foodItem) => {
        const today = new Date().toISOString().split('T')[0];

        if (today !== lastActiveDate) {
            setLastActiveDate(today);
            setWater({ current: 0, goal: water.goal || 2000, history: [] });
            setMeals({
                breakfast: [], lunch: [], snack: [], dinner: [],
                [mealType]: [foodItem]
            });
        } else {
            setMeals(prev => ({
                ...prev,
                [mealType]: [...prev[mealType], foodItem]
            }));
        }
    };

    const removeFood = (mealType, index) => {
        setMeals(prev => ({
            ...prev,
            [mealType]: prev[mealType].filter((_, i) => i !== index)
        }));
    };

    const updateFoodQuantity = (mealType, index, newQuantity) => {
        setMeals(prev => {
            const newList = [...prev[mealType]];
            const item = newList[index];

            const baseCalories = item.baseCalories || (item.quantity ? item.calories / item.quantity : item.calories);
            const baseProtein = item.baseProtein || (item.quantity ? item.protein / item.quantity : item.protein);
            const baseCarbs = item.baseCarbs || (item.quantity ? item.carbs / item.quantity : item.carbs);
            const baseFat = item.baseFat || (item.quantity ? item.fat / item.quantity : item.fat);

            newList[index] = {
                ...item,
                quantity: newQuantity,
                calories: Math.round(baseCalories * newQuantity),
                protein: (baseProtein * newQuantity).toFixed(1),
                carbs: (baseCarbs * newQuantity).toFixed(1),
                fat: (baseFat * newQuantity).toFixed(1)
            };

            return { ...prev, [mealType]: newList };
        });
    };

    const clearMeal = (mealType) => {
        setMeals(prev => ({ ...prev, [mealType]: [] }));
    };

    const updateProfile = (key, value) => {
        setUser(prev => ({ ...prev, [key]: value }));
    };

    const togglePreference = (key) => {
        setPreferences(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const completeOnboarding = (userData) => {
        setUser(userData);
        setIsOnboarded(true);
        const today = new Date().toISOString().split('T')[0];
        setLastActiveDate(today);
    };

    const resetApp = () => {
        setUser(defaultUser);
        setIsOnboarded(false);
        setMeals({ breakfast: [], lunch: [], snack: [], dinner: [] });
        setWater({ current: 0, goal: 2000, history: [] });
        setDailyStats({});
        setLastActiveDate('');
        localStorage.clear();
    };

    return (
        <UserContext.Provider value={{
            user,
            isOnboarded,
            isLoading,
            meals,
            water,
            preferences,
            dailyStats,
            addWater,
            removeWater,
            addFood,
            removeFood,
            updateFoodQuantity,
            clearMeal,
            updateProfile,
            togglePreference,
            completeOnboarding,
            resetApp
        }}>
            {children}
        </UserContext.Provider>
    );
}

export function useUser() {
    const context = useContext(UserContext);
    // Return default values if not in UserProvider (e.g., during SSR)
    if (!context) {
        return {
            user: { name: '', email: '', weight: 0, height: 0, age: 0, gender: 'male', bmi: 0, goalCalories: 0, goalWater: 0 },
            isOnboarded: false,
            isLoading: true,
            meals: { breakfast: [], lunch: [], snack: [], dinner: [] },
            water: { current: 0, goal: 2000, history: [] },
            preferences: {},
            dailyStats: {},
            addWater: () => { },
            removeWater: () => { },
            addFood: () => { },
            removeFood: () => { },
            updateFoodQuantity: () => { },
            clearMeal: () => { },
            updateProfile: () => { },
            togglePreference: () => { },
            completeOnboarding: () => { },
            resetApp: () => { }
        };
    }
    return context;
}
