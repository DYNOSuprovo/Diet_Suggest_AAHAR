"use client";
import { useState, useEffect } from 'react';
import { Capacitor } from '@capacitor/core';

// This component adds a black status bar area ONLY for Android/mobile apps
// It won't show on web browsers
export default function StatusBarSpacer() {
    const [isNative, setIsNative] = useState(false);
    const [statusBarHeight, setStatusBarHeight] = useState(0);

    useEffect(() => {
        // Check if running in Capacitor (native app)
        const native = Capacitor.isNativePlatform();
        setIsNative(native);

        if (native) {
            // Standard Android status bar height is around 24-25dp
            // On notched phones it can be higher, we use a safe value
            const platform = Capacitor.getPlatform();
            if (platform === 'android') {
                // Get the actual status bar height from CSS env variable or use default
                const safeAreaTop = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sat') || '0');
                setStatusBarHeight(safeAreaTop > 0 ? safeAreaTop : 28);
            } else if (platform === 'ios') {
                setStatusBarHeight(44); // iOS typically has 44px safe area
            }
        }
    }, []);

    // Don't render anything on web
    if (!isNative) return null;

    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                height: `${statusBarHeight}px`,
                backgroundColor: '#000000',
                zIndex: 9999,
            }}
        />
    );
}
