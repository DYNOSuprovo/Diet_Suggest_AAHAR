"use client";
import { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
    const [theme, setTheme] = useState('light');
    const [isLoading, setIsLoading] = useState(true);

    // Load theme from localStorage on mount
    useEffect(() => {
        const savedTheme = localStorage.getItem('aahar_theme');
        if (savedTheme) {
            setTheme(savedTheme);
        } else {
            // Check system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            setTheme(prefersDark ? 'dark' : 'light');
        }
        setIsLoading(false);
    }, []);

    // Apply theme to document
    useEffect(() => {
        if (!isLoading) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('aahar_theme', theme);
        }
    }, [theme, isLoading]);

    const toggleTheme = () => {
        setTheme(prev => prev === 'light' ? 'dark' : 'light');
    };

    const setLightTheme = () => setTheme('light');
    const setDarkTheme = () => setTheme('dark');

    return (
        <ThemeContext.Provider value={{
            theme,
            isDark: theme === 'dark',
            isLight: theme === 'light',
            toggleTheme,
            setLightTheme,
            setDarkTheme
        }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        return {
            theme: 'light',
            isDark: false,
            isLight: true,
            toggleTheme: () => { },
            setLightTheme: () => { },
            setDarkTheme: () => { }
        };
    }
    return context;
}
