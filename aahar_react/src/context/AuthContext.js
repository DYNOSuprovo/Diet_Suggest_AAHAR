"use client";
import { createContext, useContext, useState, useEffect } from 'react';
import {
    onAuthStateChanged,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signInWithPopup,
    signInWithCredential,
    GoogleAuthProvider,
    signOut,
    updateProfile
} from 'firebase/auth';
import { auth, googleProvider, facebookProvider } from '../lib/firebase';

const AuthContext = createContext();

// Detect if running in Capacitor (mobile app)
const isCapacitor = () => {
    return typeof window !== 'undefined' && window.Capacitor !== undefined;
};

// Dynamic import for Capacitor Firebase Auth (only on mobile)
let FirebaseAuthentication = null;
if (typeof window !== 'undefined') {
    import('@capacitor-firebase/authentication').then((module) => {
        FirebaseAuthentication = module.FirebaseAuthentication;
    }).catch(() => {
        console.log('Capacitor Firebase Auth not available');
    });
}

export function AuthProvider({ children }) {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [authError, setAuthError] = useState(null);

    // Listen to auth state changes
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            setCurrentUser(user);
            setLoading(false);
        });

        return unsubscribe;
    }, []);

    // Clear error after 5 seconds
    useEffect(() => {
        if (authError) {
            const timer = setTimeout(() => setAuthError(null), 5000);
            return () => clearTimeout(timer);
        }
    }, [authError]);

    // Email/Password Sign Up
    const signup = async (email, password, displayName) => {
        try {
            setAuthError(null);
            const result = await createUserWithEmailAndPassword(auth, email, password);
            if (displayName) {
                await updateProfile(result.user, { displayName });
            }
            return { success: true, user: result.user };
        } catch (error) {
            const errorMessage = getErrorMessage(error.code);
            setAuthError(errorMessage);
            return { success: false, error: errorMessage };
        }
    };

    // Email/Password Sign In
    const login = async (email, password) => {
        try {
            setAuthError(null);
            const result = await signInWithEmailAndPassword(auth, email, password);
            return { success: true, user: result.user };
        } catch (error) {
            const errorMessage = getErrorMessage(error.code);
            setAuthError(errorMessage);
            return { success: false, error: errorMessage };
        }
    };

    // Google Sign In - Uses native plugin for mobile, popup for web
    const loginWithGoogle = async () => {
        try {
            setAuthError(null);

            // For Capacitor/mobile apps, use native Google Sign-In
            if (isCapacitor() && FirebaseAuthentication) {
                try {
                    // Sign in with native Google
                    const result = await FirebaseAuthentication.signInWithGoogle();

                    // Get the ID token from native sign-in
                    if (result.credential?.idToken) {
                        // Create Firebase credential and sign in
                        const credential = GoogleAuthProvider.credential(result.credential.idToken);
                        const userCredential = await signInWithCredential(auth, credential);
                        return { success: true, user: userCredential.user };
                    } else {
                        throw new Error('No ID token received');
                    }
                } catch (nativeError) {
                    console.error('Native Google Sign-In error:', nativeError);
                    // If native fails, show error message
                    setAuthError('Google Sign-In failed. Please try email/password login.');
                    return { success: false, error: nativeError.message };
                }
            } else {
                // For web, use popup
                const result = await signInWithPopup(auth, googleProvider);
                return { success: true, user: result.user };
            }
        } catch (error) {
            const errorMessage = getErrorMessage(error.code);
            setAuthError(errorMessage);
            return { success: false, error: errorMessage };
        }
    };

    // Facebook Sign In - popup for web, message for mobile
    const loginWithFacebook = async () => {
        try {
            setAuthError(null);

            // For Capacitor/mobile apps, Facebook needs additional setup
            if (isCapacitor()) {
                setAuthError('Facebook Sign-In is not available in the app. Please use email and password or Google.');
                return { success: false, error: 'Use email login in app' };
            } else {
                const result = await signInWithPopup(auth, facebookProvider);
                return { success: true, user: result.user };
            }
        } catch (error) {
            const errorMessage = getErrorMessage(error.code);
            setAuthError(errorMessage);
            return { success: false, error: errorMessage };
        }
    };

    // Sign Out
    const logout = async () => {
        try {
            setAuthError(null);

            // Also sign out from native auth if in Capacitor
            if (isCapacitor() && FirebaseAuthentication) {
                try {
                    await FirebaseAuthentication.signOut();
                } catch (e) {
                    console.log('Native sign out error:', e);
                }
            }

            await signOut(auth);
            return { success: true };
        } catch (error) {
            const errorMessage = getErrorMessage(error.code);
            setAuthError(errorMessage);
            return { success: false, error: errorMessage };
        }
    };

    // Helper to get user-friendly error messages
    const getErrorMessage = (errorCode) => {
        switch (errorCode) {
            case 'auth/email-already-in-use':
                return 'This email is already registered. Please login instead.';
            case 'auth/invalid-email':
                return 'Please enter a valid email address.';
            case 'auth/weak-password':
                return 'Password should be at least 6 characters.';
            case 'auth/user-not-found':
                return 'No account found with this email.';
            case 'auth/wrong-password':
                return 'Incorrect password. Please try again.';
            case 'auth/invalid-credential':
                return 'Invalid email or password.';
            case 'auth/too-many-requests':
                return 'Too many failed attempts. Please try again later.';
            case 'auth/popup-closed-by-user':
                return 'Sign-in popup was closed. Please try again.';
            case 'auth/cancelled-popup-request':
                return 'Sign-in was cancelled.';
            case 'auth/account-exists-with-different-credential':
                return 'An account already exists with this email using a different sign-in method.';
            case 'auth/unauthorized-domain':
                return 'This domain is not authorized. Please use email/password login.';
            default:
                return 'An error occurred. Please try again.';
        }
    };

    const value = {
        currentUser,
        loading,
        authError,
        signup,
        login,
        loginWithGoogle,
        loginWithFacebook,
        logout,
        clearError: () => setAuthError(null),
        isCapacitor: isCapacitor()
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    // Return default values if not in AuthProvider (e.g., during SSR)
    if (!context) {
        return {
            currentUser: null,
            loading: false,
            authError: null,
            signup: async () => ({ success: false }),
            login: async () => ({ success: false }),
            loginWithGoogle: async () => ({ success: false }),
            loginWithFacebook: async () => ({ success: false }),
            logout: async () => ({ success: false }),
            clearError: () => { },
            isCapacitor: false
        };
    }
    return context;
}
