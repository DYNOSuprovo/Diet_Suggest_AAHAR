"use client";
import { initializeApp, getApps } from 'firebase/app';
import { getAuth, GoogleAuthProvider, FacebookAuthProvider } from 'firebase/auth';
import { getDatabase, ref, set, get, push, serverTimestamp } from 'firebase/database';

const firebaseConfig = {
    apiKey: "AIzaSyBEUVu8WdHVcF1c9JrfEgfMYvoxpwuaAgE",
    authDomain: "aaharai.firebaseapp.com",
    projectId: "aaharai",
    storageBucket: "aaharai.firebasestorage.app",
    messagingSenderId: "560546432733",
    appId: "1:560546432733:web:d0f67d197e8970c5677d34",
    databaseURL: "https://aaharai-default-rtdb.asia-southeast1.firebasedatabase.app"
};

// Initialize Firebase only if not already initialized
const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];

export const auth = getAuth(app);
export const database = getDatabase(app);
export const googleProvider = new GoogleAuthProvider();
export const facebookProvider = new FacebookAuthProvider();

// Configure Google provider for better UX
googleProvider.setCustomParameters({
    prompt: 'select_account'
});

// Realtime Database helper functions
export const saveUserData = async (userId, data) => {
    try {
        const userRef = ref(database, `users/${userId}`);
        await set(userRef, {
            ...data,
            updatedAt: Date.now()
        });
        return { success: true };
    } catch (error) {
        console.error('Error saving user data:', error);
        return { success: false, error };
    }
};

export const getUserData = async (userId) => {
    try {
        const userRef = ref(database, `users/${userId}`);
        const snapshot = await get(userRef);
        if (snapshot.exists()) {
            return { success: true, data: snapshot.val() };
        }
        return { success: true, data: null };
    } catch (error) {
        console.error('Error getting user data:', error);
        return { success: false, error };
    }
};

export const saveFeedback = async (userId, feedback) => {
    try {
        const feedbackRef = ref(database, 'feedback');
        const newFeedbackRef = push(feedbackRef);
        await set(newFeedbackRef, {
            userId: userId || 'anonymous',
            ...feedback,
            createdAt: Date.now(),
            resolved: false
        });
        return { success: true };
    } catch (error) {
        console.error('Error saving feedback:', error);
        return { success: false, error };
    }
};
