// Analytics utility functions
// This tracks anonymous usage data to improve the app

export const GA_TRACKING_ID = process.env.NEXT_PUBLIC_GA_ID || '';

// Track page views
export const pageview = (url) => {
    if (typeof window !== 'undefined' && window.gtag) {
        window.gtag('config', GA_TRACKING_ID, {
            page_path: url,
        });
    }
};

// Track events
export const event = ({ action, category, label, value }) => {
    if (typeof window !== 'undefined' && window.gtag) {
        window.gtag('event', action, {
            event_category: category,
            event_label: label,
            value: value,
        });
    }
};

// Track feature usage
export const trackFeatureUse = (featureName) => {
    event({
        action: 'feature_used',
        category: 'engagement',
        label: featureName,
    });
};

// Track food search
export const trackFoodSearch = (query, resultsCount) => {
    event({
        action: 'food_search',
        category: 'search',
        label: query,
        value: resultsCount,
    });
};

// Track dietary preference change
export const trackPreferenceChange = (preference, enabled) => {
    event({
        action: 'preference_changed',
        category: 'settings',
        label: `${preference}_${enabled ? 'on' : 'off'}`,
    });
};

// Track meal added
export const trackMealAdded = (mealType, calories) => {
    event({
        action: 'meal_added',
        category: 'tracking',
        label: mealType,
        value: Math.round(calories),
    });
};

// Track AI chat interaction
export const trackChatInteraction = (messageLength) => {
    event({
        action: 'chat_message',
        category: 'ai_assistant',
        label: 'chat_sent',
        value: messageLength,
    });
};
