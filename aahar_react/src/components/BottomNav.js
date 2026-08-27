"use client";
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { Home, MessageSquare, Droplets, UtensilsCrossed, User } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { useLanguage } from '../context/LanguageContext';

export default function BottomNav() {
    const pathname = usePathname();
    const { isDark } = useTheme();
    const { t } = useLanguage();

    // Hide on certain pages
    if (['/', '/login', '/signup', '/onboarding'].includes(pathname)) return null;

    const navItems = [
        { icon: Home, path: '/dashboard', labelKey: 'nav_home' },
        { icon: MessageSquare, path: '/chat', labelKey: 'nav_chat' },
        { icon: Droplets, path: '/water', labelKey: 'nav_water' },
        { icon: UtensilsCrossed, path: '/mess', labelKey: 'nav_mess' },
        { icon: User, path: '/profile', labelKey: 'nav_profile' },
    ];

    const activeIndex = navItems.findIndex(item => item.path === pathname);

    return (
        <motion.div
            initial={{ y: 100 }}
            animate={{ y: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            style={{
                position: 'fixed',
                bottom: 0,
                left: 0,
                right: 0,
                background: isDark
                    ? 'linear-gradient(180deg, rgba(30, 30, 30, 0.4) 0%, rgba(20, 20, 20, 0.6) 100%)'
                    : 'linear-gradient(180deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.5) 100%)',
                backdropFilter: 'saturate(200%) blur(40px)',
                WebkitBackdropFilter: 'saturate(200%) blur(40px)',
                borderTop: isDark
                    ? '0.5px solid rgba(255, 255, 255, 0.1)'
                    : '0.5px solid rgba(255, 255, 255, 0.3)',
                boxShadow: isDark
                    ? '0 -1px 0 rgba(0,0,0,0.3)'
                    : '0 -1px 0 rgba(0,0,0,0.02)',
                zIndex: 9999
            }}
        >
            <div style={{
                maxWidth: '480px',
                margin: '0 auto',
                padding: '8px 16px 20px 16px',
                position: 'relative'
            }}>
                {/* Sliding Indicator */}
                <motion.div
                    style={{
                        position: 'absolute',
                        bottom: '8px',
                        height: '3px',
                        width: '48px',
                        backgroundColor: '#1DB954',
                        borderRadius: '4px'
                    }}
                    animate={{
                        left: `calc(${activeIndex * 20}% + 10% - 24px)`
                    }}
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />

                <div style={{
                    display: 'flex',
                    justifyContent: 'space-around',
                    alignItems: 'center'
                }}>
                    {navItems.map((item, index) => {
                        const Icon = item.icon;
                        const isActive = pathname === item.path;

                        return (
                            <Link
                                key={item.path}
                                href={item.path}
                                style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    alignItems: 'center',
                                    padding: '8px 12px',
                                    textDecoration: 'none'
                                }}
                            >
                                <motion.div
                                    animate={{
                                        scale: isActive ? 1.15 : 1,
                                        y: isActive ? -2 : 0
                                    }}
                                    transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                                >
                                    <Icon
                                        size={24}
                                        color={isActive ? '#1DB954' : (isDark ? '#6b7280' : '#9CA3AF')}
                                        strokeWidth={isActive ? 2.5 : 2}
                                    />
                                </motion.div>
                                <motion.span
                                    animate={{
                                        opacity: isActive ? 1 : 0.6,
                                        y: isActive ? 0 : 2
                                    }}
                                    style={{
                                        fontSize: '10px',
                                        marginTop: '4px',
                                        fontWeight: isActive ? '600' : '500',
                                        color: isActive ? '#1DB954' : (isDark ? '#9ca3af' : '#9CA3AF')
                                    }}
                                >
                                    {t(item.labelKey)}
                                </motion.span>
                            </Link>
                        );
                    })}
                </div>
            </div>
        </motion.div>
    );
}
