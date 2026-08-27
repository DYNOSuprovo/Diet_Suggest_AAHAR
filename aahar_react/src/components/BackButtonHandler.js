"use client";
import { useEffect, useRef } from 'react';

import { App } from '@capacitor/app';
import { useRouter, usePathname } from 'next/navigation';

const BackButtonHandler = () => {
    const router = useRouter();
    const pathname = usePathname();
    const pathnameRef = useRef(pathname);

    // 1. Keep the ref updated with the latest pathname
    useEffect(() => {
        pathnameRef.current = pathname;
        console.log('BackButtonHandler: Path updated to', pathname);
    }, [pathname]);

    // 2. Register logic ONCE
    useEffect(() => {
        let backListenerShim = null; // Store the listener handle

        const setupListener = async () => {
            console.log('BackButtonHandler: Registering listener...');
            backListenerShim = await App.addListener('backButton', (event) => {
                // Remove Ionic-specific event.detail.register which crashes standard Capacitor apps

                // We must use the Ref to get the LATEST path inside this closure!
                const currentPath = pathnameRef.current;
                console.log('BackButtonHandler: Event fired. Current path ref:', currentPath);

                // 1. Pages that should EXIT the app
                const exitPages = [
                    '/login',
                    '/onboarding',
                    '/dashboard',
                    '/chat',
                    '/mess',
                    '/water',
                    '/profile', // Base profile page
                ];

                // Normalizing path to handle query strings or trailing slashes
                // e.g. /profile?foo=bar -> /profile
                const normalizedPath = currentPath.split('?')[0].replace(/\/$/, '') || '/';

                if (normalizedPath === '/' || exitPages.includes(normalizedPath)) {
                    console.log('BackButtonHandler: Exiting app (Root/Tab page)');
                    App.exitApp();
                }
                else if (normalizedPath.includes('about') || normalizedPath.includes('privacy')) {
                    // Explicit redirect as requested
                    console.log('BackButtonHandler: Redirecting to profile');
                    router.replace('/profile');
                }
                else {
                    console.log('BackButtonHandler: Attempting router.back()');
                    // router.back() integrates better with Next.js client-side routing
                    router.back();
                }
            });
        };

        setupListener();

        // Cleanup function
        return () => {
            console.log('BackButtonHandler: Removing listener');
            if (backListenerShim) {
                backListenerShim.remove();
            } else {
                // Fallback if the promise hadn't resolved yet (rare but possible in strict mode double mounting)
                App.removeAllListeners();
            }
        };
    }, []); // Empty dependency array = Runs once on mount!

    return null;
};

export default BackButtonHandler;
