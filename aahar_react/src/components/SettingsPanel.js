"use client";
import { useState } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { Moon, Sun, Globe, ChevronRight, X, Check } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { useLanguage } from '../context/LanguageContext';
import { useUser } from '../context/UserContext';

// Dietary Toggle Component
const DietaryToggle = ({ label, prefKey }) => {
    const { preferences, togglePreference } = useUser();
    const isActive = preferences[prefKey];

    return (
        <motion.div
            whileTap={{ scale: 0.98 }}
            onClick={() => togglePreference(prefKey)}
            style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '14px 0',
                borderBottom: '1px solid var(--border-color)',
                cursor: 'pointer'
            }}
        >
            <span style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>
                {label}
            </span>
            <div style={{
                width: '50px',
                height: '28px',
                borderRadius: '14px',
                background: isActive ? '#1DB954' : '#e5e7eb',
                padding: '3px',
                cursor: 'pointer',
                transition: 'background 0.3s'
            }}>
                <motion.div
                    animate={{ x: isActive ? 22 : 0 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    style={{
                        width: '22px',
                        height: '22px',
                        borderRadius: '11px',
                        background: 'white',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                    }}
                />
            </div>
        </motion.div>
    );
};

export default function SettingsPanel() {
    const { isDark, toggleTheme } = useTheme();
    const { language, setLanguage, languages, currentLanguageName, t } = useLanguage();
    const [showLanguageModal, setShowLanguageModal] = useState(false);

    return (
        <div style={{
            background: 'var(--bg-card)',
            borderRadius: '20px',
            padding: '20px',
            boxShadow: 'var(--shadow-sm)'
        }}>
            <h3 style={{
                fontSize: '16px',
                fontWeight: '600',
                color: 'var(--text-primary)',
                marginBottom: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
            }}>
                ⚙️ {t('profile_settings')}
            </h3>

            {/* Dark Mode Toggle */}
            <motion.div
                whileTap={{ scale: 0.98 }}
                onClick={toggleTheme}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '14px 16px',
                    background: 'var(--bg-secondary)',
                    borderRadius: '14px',
                    marginBottom: '12px',
                    cursor: 'pointer',
                    border: '1px solid var(--border-color)'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '12px',
                        background: isDark ? 'linear-gradient(135deg, #1e293b 0%, #334155 100%)' : 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        {isDark ? <Moon size={20} color="#a5b4fc" /> : <Sun size={20} color="#f59e0b" />}
                    </div>
                    <div>
                        <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)' }}>
                            {t('profile_theme')}
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                            {isDark ? 'Dark' : 'Light'}
                        </div>
                    </div>
                </div>

                {/* Toggle Switch */}
                <div style={{
                    width: '50px',
                    height: '28px',
                    borderRadius: '14px',
                    background: isDark ? '#1DB954' : '#e5e7eb',
                    padding: '3px',
                    cursor: 'pointer',
                    transition: 'background 0.3s'
                }}>
                    <motion.div
                        animate={{ x: isDark ? 22 : 0 }}
                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        style={{
                            width: '22px',
                            height: '22px',
                            borderRadius: '11px',
                            background: 'white',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                        }}
                    />
                </div>
            </motion.div>

            {/* Language Selector */}
            <motion.div
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowLanguageModal(true)}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '14px 16px',
                    background: 'var(--bg-secondary)',
                    borderRadius: '14px',
                    cursor: 'pointer',
                    border: '1px solid var(--border-color)'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '12px',
                        background: 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        <Globe size={20} color="#3b82f6" />
                    </div>
                    <div>
                        <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)' }}>
                            {t('profile_language')}
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                            {currentLanguageName}
                        </div>
                    </div>
                </div>

                <ChevronRight size={20} color="var(--text-muted)" />
            </motion.div>

            {/* Dietary Preferences Section */}
            <div style={{
                background: 'var(--bg-card)',
                borderRadius: '20px',
                padding: '20px',
                marginTop: '20px',
                boxShadow: 'var(--shadow-sm)'
            }}>
                <h3 style={{
                    fontSize: '16px',
                    fontWeight: '600',
                    color: 'var(--text-primary)',
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                }}>
                    🥗 Dietary Preferences
                </h3>

                <DietaryToggle label="Vegetarian" prefKey="vegetarian" />
                <DietaryToggle label="Gluten Free" prefKey="glutenFree" />
                <DietaryToggle label="Dairy Free" prefKey="dairyFree" />
                <DietaryToggle label="Low Carb" prefKey="lowCarb" />
            </div>

            {/* Links Section */}
            <div style={{
                background: 'var(--bg-card)',
                borderRadius: '20px',
                padding: '20px',
                marginTop: '20px',
                boxShadow: 'var(--shadow-sm)'
            }}>
                <Link href="/privacy" style={{ textDecoration: 'none' }}>
                    <motion.div
                        whileTap={{ scale: 0.98 }}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '14px 16px',
                            background: 'var(--bg-secondary)',
                            borderRadius: '14px',
                            marginBottom: '12px',
                            cursor: 'pointer',
                            border: '1px solid var(--border-color)'
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={{
                                width: '40px',
                                height: '40px',
                                borderRadius: '12px',
                                background: 'linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}>
                                🛡️
                            </div>
                            <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)' }}>
                                Privacy Policy
                            </div>
                        </div>
                        <ChevronRight size={20} color="var(--text-muted)" />
                    </motion.div>
                </Link>

                <Link href="/about" style={{ textDecoration: 'none' }}>
                    <motion.div
                        whileTap={{ scale: 0.98 }}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '14px 16px',
                            background: 'var(--bg-secondary)',
                            borderRadius: '14px',
                            cursor: 'pointer',
                            border: '1px solid var(--border-color)'
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={{
                                width: '40px',
                                height: '40px',
                                borderRadius: '12px',
                                background: 'linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}>
                                ℹ️
                            </div>
                            <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)' }}>
                                About Aahar
                            </div>
                        </div>
                        <ChevronRight size={20} color="var(--text-muted)" />
                    </motion.div>
                </Link>
            </div>

            {/* Language Selection Modal */}
            <AnimatePresence>
                {showLanguageModal && (
                    <>
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setShowLanguageModal(false)}
                            style={{
                                position: 'fixed',
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                background: 'rgba(0,0,0,0.5)',
                                zIndex: 10000,
                                backdropFilter: 'blur(4px)'
                            }}
                        />
                        <motion.div
                            initial={{ y: '100%' }}
                            animate={{ y: 0 }}
                            exit={{ y: '100%' }}
                            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                            style={{
                                position: 'fixed',
                                bottom: 0,
                                left: 0,
                                right: 0,
                                background: 'var(--bg-card)',
                                borderTopLeftRadius: '24px',
                                borderTopRightRadius: '24px',
                                padding: '24px',
                                zIndex: 10001,
                                maxHeight: '80vh',
                                overflowY: 'auto'
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: 'var(--text-primary)' }}>Select Language</h3>
                                <button
                                    onClick={() => setShowLanguageModal(false)}
                                    style={{
                                        border: 'none',
                                        background: 'var(--bg-secondary)',
                                        borderRadius: '50%',
                                        width: '32px',
                                        height: '32px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        cursor: 'pointer'
                                    }}
                                >
                                    <X size={18} color="var(--text-primary)" />
                                </button>
                            </div>

                            <div style={{ display: 'grid', gap: '12px' }}>
                                {languages.map((lang) => (
                                    <motion.button
                                        key={lang.code}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={() => {
                                            setLanguage(lang.code);
                                            setShowLanguageModal(false);
                                        }}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            padding: '16px',
                                            borderRadius: '16px',
                                            background: language === lang.code ? 'var(--accent-light)' : 'var(--bg-secondary)',
                                            border: language === lang.code ? '1px solid var(--accent-primary)' : '1px solid transparent',
                                            color: 'var(--text-primary)',
                                            fontSize: '16px',
                                            fontWeight: '500',
                                            cursor: 'pointer',
                                            textAlign: 'left'
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                            <span style={{ fontSize: '20px' }}>
                                                {lang.code === 'en' ? '🇺🇸' : '🇮🇳'}
                                            </span>
                                            {lang.name}
                                        </div>
                                        {language === lang.code && (
                                            <Check size={20} color="var(--accent-primary)" />
                                        )}
                                    </motion.button>
                                ))}
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </div>
    );
}
