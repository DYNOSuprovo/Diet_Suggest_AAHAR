const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:10000/api";

// Cache for nutrition data
let nutritionDataCache = null;

export const getSessionId = () => {
    if (typeof window === 'undefined') return null;
    let sessionId = localStorage.getItem('aahar_session_id');
    if (!sessionId) {
        sessionId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        localStorage.setItem('aahar_session_id', sessionId);
    }
    return sessionId;
};

export const sendMessageToBackend = async (message) => {
    const sessionId = getSessionId();
    let userProfile = null;
    try {
        const savedUser = localStorage.getItem('aahar_user');
        if (savedUser) {
            userProfile = JSON.parse(savedUser);
        }
    } catch (e) {
        console.error("Could not parse user profile from localStorage", e);
    }

    try {
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: message,
                session_id: sessionId,
                user_profile: userProfile
            }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        return data.answer || data.response || JSON.stringify(data);
    } catch (error) {
        console.error("API Error:", error);
        throw error;
    }
};

export const getChatHistory = async () => {
    const sessionId = getSessionId();
    try {
        const response = await fetch(`${BACKEND_URL}/history?session_id=${sessionId}`);
        if (!response.ok) return [];
        const data = await response.json();
        return data.history || [];
    } catch (error) {
        console.error("History Error:", error);
        return [];
    }
};

// Load nutrition data from local JSON file
const loadNutritionData = async () => {
    if (nutritionDataCache) {
        return nutritionDataCache;
    }

    try {
        const response = await fetch('/nutrition_data.json');
        if (!response.ok) {
            console.error('Failed to load nutrition data');
            return [];
        }
        nutritionDataCache = await response.json();
        return nutritionDataCache;
    } catch (error) {
        console.error('Error loading nutrition data:', error);
        return [];
    }
};

export const searchFood = async (query) => {
    if (!query) return [];

    try {
        // Load nutrition data
        const nutritionData = await loadNutritionData();

        if (!nutritionData || nutritionData.length === 0) {
            return [];
        }

        // Perform case-insensitive search
        const searchTerm = query.toLowerCase().trim();
        const results = nutritionData.filter(item => {
            const dishName = item["Dish Name"]?.toLowerCase() || '';
            const category = item["Category"]?.toLowerCase() || '';
            const region = item["Region"]?.toLowerCase() || '';

            return dishName.includes(searchTerm) ||
                category.includes(searchTerm) ||
                region.includes(searchTerm);
        });

        // Return top 100 results for better discoverability
        return results.slice(0, 100);
    } catch (error) {
        console.error("Search Error:", error);
        return [];
    }
};

export const analyzeMeal = async (dishNames) => {
    try {
        const response = await fetch(`${BACKEND_URL}/analyze-meal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dish_names: dishNames })
        });
        if (!response.ok) throw new Error("Analysis failed");
        return await response.json();
    } catch (error) {
        console.error("Analysis Error:", error);
        throw error;
    }
};
