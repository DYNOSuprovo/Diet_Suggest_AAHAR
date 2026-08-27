"use client";
import { createContext, useContext, useState, useEffect } from 'react';
import { translations } from '../lib/translations';

const LanguageContext = createContext();

export function LanguageProvider({ children }) {
    const [language, setLanguage] = useState('en');
    const [isLoading, setIsLoading] = useState(true);

    // Load language from localStorage on mount
    useEffect(() => {
        const savedLang = localStorage.getItem('aahar_language');
        if (savedLang && translations[savedLang]) {
            setLanguage(savedLang);
        }
        setIsLoading(false);
    }, []);

    // Save language preference
    useEffect(() => {
        if (!isLoading) {
            localStorage.setItem('aahar_language', language);
        }
    }, [language, isLoading]);

    // Translation function
    const t = (key) => {
        return translations[language]?.[key] || translations['en']?.[key] || key;
    };

    const currentLanguageName = translations[language]?.name || 'English';

    return (
        <LanguageContext.Provider value={{
            language,
            setLanguage,
            languages: Object.keys(translations).map(key => ({ code: key, name: translations[key].name })),
            currentLanguageName,
            t
        }}>
            {children}
        </LanguageContext.Provider>
    );
}

export function useLanguage() {
    const context = useContext(LanguageContext);
    if (!context) {
        return {
            language: 'en',
            setLanguage: () => { },
            languages: [{ code: 'en', name: 'English' }],
            currentLanguageName: 'English',
            t: (key) => key
        };
    }
    return context;
}
