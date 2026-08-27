"use client";
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Lock, Eye, EyeOff, User, ArrowRight, Sparkles, AlertCircle, Loader2 } from 'lucide-react';
import { useUser } from '../../context/UserContext';
import { useAuth } from '../../context/AuthContext';

// Floating food icons component
const FloatingIcons = () => {
    const foods = ['🥗', '🍎', '🥑', '🍳', '🥕', '🍊', '🥬', '🍇'];
    return (
        <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}>
            {foods.map((food, i) => (
                <motion.span
                    key={i}
                    initial={{
                        x: Math.random() * 400,
                        y: -50,
                        opacity: 0.6,
                        rotate: 0
                    }}
                    animate={{
                        y: '100vh',
                        rotate: 360,
                        opacity: [0.6, 0.8, 0.6]
                    }}
                    transition={{
                        duration: 10 + Math.random() * 10,
                        repeat: Infinity,
                        delay: i * 1.5,
                        ease: "linear"
                    }}
                    style={{
                        position: 'absolute',
                        fontSize: '32px',
                        left: `${(i * 12) + 5}%`
                    }}
                >
                    {food}
                </motion.span>
            ))}
        </div>
    );
};

// Password strength indicator
const PasswordStrength = ({ password }) => {
    const getStrength = () => {
        if (!password) return 0;
        let strength = 0;
        if (password.length >= 6) strength++;
        if (password.length >= 8) strength++;
        if (/[A-Z]/.test(password)) strength++;
        if (/[0-9]/.test(password)) strength++;
        if (/[^A-Za-z0-9]/.test(password)) strength++;
        return strength;
    };

    const strength = getStrength();
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#1DB954'];
    const labels = ['Weak', 'Fair', 'Good', 'Strong', 'Excellent'];

    if (!password) return null;

    return (
        <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            style={{ marginTop: '8px' }}
        >
            <div style={{ display: 'flex', gap: '4px', marginBottom: '4px' }}>
                {[1, 2, 3, 4, 5].map((i) => (
                    <motion.div
                        key={i}
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: strength >= i ? 1 : 0.3 }}
                        transition={{ delay: i * 0.1 }}
                        style={{
                            flex: 1,
                            height: '4px',
                            borderRadius: '2px',
                            background: strength >= i ? colors[strength - 1] : '#e0e0e0',
                            transformOrigin: 'left'
                        }}
                    />
                ))}
            </div>
            <span style={{ fontSize: '11px', color: colors[strength - 1] || '#999' }}>
                {labels[strength - 1] || ''}
            </span>
        </motion.div>
    );
};


export default function Login() {
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [focusedInput, setFocusedInput] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const router = useRouter();
    const { isOnboarded, updateProfile } = useUser();
    const { login, signup, loginWithGoogle, loginWithFacebook, authError, clearError } = useAuth();

    const handleNavigation = (user) => {
        if (user) {
            if (user.email) updateProfile('email', user.email);
            if (user.displayName) updateProfile('name', user.displayName);
        }

        if (isOnboarded) {
            router.push('/dashboard');
        } else {
            router.push('/onboarding');
        }
    };

    const handleAuth = async (e) => {
        if (e) e.preventDefault();
        if (isSubmitting) return;

        setIsSubmitting(true);
        clearError();

        let result;
        if (isLogin) {
            result = await login(email, password);
        } else {
            const displayName = name || email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1);
            result = await signup(email, password, displayName);
        }

        setIsSubmitting(false);

        if (result.success) {
            handleNavigation(result.user);
        }
    };

    const handleGoogleSignIn = async () => {
        if (isSubmitting) return;
        setIsSubmitting(true);
        clearError();

        const result = await loginWithGoogle();
        setIsSubmitting(false);

        if (result.success) {
            handleNavigation(result.user);
        }
    };

    const handleFacebookSignIn = async () => {
        if (isSubmitting) return;
        setIsSubmitting(true);
        clearError();

        const result = await loginWithFacebook();
        setIsSubmitting(false);

        if (result.success) {
            handleNavigation(result.user);
        }
    };

    const handleGuestMode = () => {
        if (isOnboarded) {
            router.push('/dashboard');
        } else {
            router.push('/onboarding');
        }
    };

    const inputStyle = (isFocused) => ({
        width: '100%',
        padding: '16px 16px 16px 48px',
        borderRadius: '16px',
        border: `2px solid ${isFocused ? '#1DB954' : 'rgba(255,255,255,0.2)'}`,
        background: 'rgba(255,255,255,0.1)',
        backdropFilter: 'blur(10px)',
        fontSize: '16px',
        color: '#333',
        outline: 'none',
        transition: 'all 0.3s ease',
        boxShadow: isFocused ? '0 0 0 4px rgba(29, 185, 84, 0.1)' : 'none'
    });

    return (
        <div style={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            background: 'linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%)',
            position: 'relative',
            overflow: 'hidden'
        }}>

            <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '40px 24px',
                position: 'relative',
                zIndex: 1
            }}>
                {/* Logo */}
                <motion.div
                    initial={{ y: -30, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 200 }}
                    style={{ marginBottom: '32px', textAlign: 'center' }}
                >
                    <h1 style={{
                        fontSize: '36px',
                        fontWeight: '800',
                        color: '#1DB954',
                        fontFamily: 'serif'
                    }}>
                        Aahar
                    </h1>
                    <p style={{ color: '#666', fontSize: '14px' }}>
                        Your journey to a healthier you
                    </p>
                </motion.div>

                {/* Blended Gradient Card */}
                <motion.div
                    initial={{ y: 40, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.2, type: "spring", stiffness: 100 }}
                    style={{
                        width: '100%',
                        maxWidth: '400px',
                        background: 'linear-gradient(180deg, rgba(255,255,255,0.85) 0%, rgba(232,245,233,0.9) 50%, rgba(200,230,201,0.85) 100%)',
                        backdropFilter: 'blur(20px)',
                        borderRadius: '32px',
                        padding: '32px 24px',
                        boxShadow: '0 8px 32px rgba(29, 185, 84, 0.08), inset 0 1px 0 rgba(255,255,255,0.6)',
                        border: 'none'
                    }}
                >
                    {/* Toggle */}
                    <div style={{
                        background: 'rgba(0,0,0,0.05)',
                        borderRadius: '16px',
                        padding: '4px',
                        display: 'flex',
                        marginBottom: '28px',
                        position: 'relative'
                    }}>
                        <motion.div
                            layout
                            style={{
                                position: 'absolute',
                                top: '4px',
                                left: isLogin ? '4px' : 'calc(50% + 2px)',
                                width: 'calc(50% - 6px)',
                                height: 'calc(100% - 8px)',
                                background: 'linear-gradient(135deg, #1DB954 0%, #0d7a3a 100%)',
                                borderRadius: '12px',
                                boxShadow: '0 4px 12px rgba(29, 185, 84, 0.3)'
                            }}
                            transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        />
                        <button
                            onClick={() => { setIsLogin(true); clearError(); }}
                            style={{
                                flex: 1,
                                padding: '12px',
                                border: 'none',
                                background: 'transparent',
                                color: isLogin ? 'white' : '#666',
                                fontWeight: '600',
                                cursor: 'pointer',
                                position: 'relative',
                                zIndex: 1,
                                transition: 'color 0.3s'
                            }}
                        >
                            Login
                        </button>
                        <button
                            onClick={() => { setIsLogin(false); clearError(); }}
                            style={{
                                flex: 1,
                                padding: '12px',
                                border: 'none',
                                background: 'transparent',
                                color: !isLogin ? 'white' : '#666',
                                fontWeight: '600',
                                cursor: 'pointer',
                                position: 'relative',
                                zIndex: 1,
                                transition: 'color 0.3s'
                            }}
                        >
                            Sign Up
                        </button>
                    </div>

                    {/* Error Message */}
                    {authError && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            style={{
                                padding: '12px 16px',
                                marginBottom: '16px',
                                background: '#FEE2E2',
                                borderRadius: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px',
                                color: '#DC2626'
                            }}
                        >
                            <AlertCircle size={20} />
                            <span style={{ fontSize: '14px' }}>{authError}</span>
                        </motion.div>
                    )}

                    <AnimatePresence mode="wait">
                        <motion.form
                            key={isLogin ? 'login' : 'signup'}
                            initial={{ opacity: 0, x: isLogin ? -20 : 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: isLogin ? 20 : -20 }}
                            transition={{ duration: 0.2 }}
                            onSubmit={handleAuth}
                        >
                            {/* Name field for signup */}
                            {!isLogin && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    style={{ marginBottom: '16px', position: 'relative' }}
                                >
                                    <User size={20} color="#999" style={{ position: 'absolute', left: '16px', top: '18px' }} />
                                    <input
                                        type="text"
                                        placeholder="Full Name"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        onFocus={() => setFocusedInput('name')}
                                        onBlur={() => setFocusedInput(null)}
                                        style={inputStyle(focusedInput === 'name')}
                                        disabled={isSubmitting}
                                    />
                                </motion.div>
                            )}

                            {/* Email */}
                            <div style={{ marginBottom: '16px', position: 'relative' }}>
                                <Mail size={20} color="#999" style={{ position: 'absolute', left: '16px', top: '18px' }} />
                                <input
                                    type="email"
                                    placeholder="Email Address"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    onFocus={() => setFocusedInput('email')}
                                    onBlur={() => setFocusedInput(null)}
                                    style={inputStyle(focusedInput === 'email')}
                                    required
                                    disabled={isSubmitting}
                                />
                            </div>

                            {/* Password */}
                            <div style={{ marginBottom: '8px', position: 'relative' }}>
                                <Lock size={20} color="#999" style={{ position: 'absolute', left: '16px', top: '18px' }} />
                                <input
                                    type={showPassword ? "text" : "password"}
                                    placeholder="Password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    onFocus={() => setFocusedInput('password')}
                                    onBlur={() => setFocusedInput(null)}
                                    style={{ ...inputStyle(focusedInput === 'password'), paddingRight: '48px' }}
                                    required
                                    minLength={6}
                                    disabled={isSubmitting}
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    style={{
                                        position: 'absolute',
                                        right: '16px',
                                        top: '18px',
                                        background: 'none',
                                        border: 'none',
                                        cursor: 'pointer',
                                        color: '#999'
                                    }}
                                >
                                    {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                                </button>
                            </div>

                            {/* Password Strength for Signup */}
                            {!isLogin && <PasswordStrength password={password} />}

                            {/* Forgot Password */}
                            {isLogin && (
                                <div style={{ textAlign: 'right', marginBottom: '24px', marginTop: '12px' }}>
                                    <a href="#" style={{ color: '#1DB954', fontSize: '14px', fontWeight: '500' }}>
                                        Forgot Password?
                                    </a>
                                </div>
                            )}

                            {/* Submit Button */}
                            <motion.button
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                type="submit"
                                disabled={isSubmitting}
                                style={{
                                    width: '100%',
                                    padding: '16px',
                                    background: 'linear-gradient(135deg, #1DB954 0%, #0d7a3a 100%)',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '16px',
                                    fontSize: '16px',
                                    fontWeight: '700',
                                    cursor: isSubmitting ? 'not-allowed' : 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '8px',
                                    boxShadow: '0 4px 20px rgba(29, 185, 84, 0.3)',
                                    marginTop: !isLogin ? '24px' : '0',
                                    opacity: isSubmitting ? 0.8 : 1
                                }}
                            >
                                {isSubmitting ? (
                                    <>
                                        <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                                        {isLogin ? 'Logging in...' : 'Creating Account...'}
                                    </>
                                ) : (
                                    <>
                                        {isLogin ? 'Login' : 'Create Account'}
                                        <ArrowRight size={18} />
                                    </>
                                )}
                            </motion.button>
                        </motion.form>
                    </AnimatePresence>

                    {/* Divider */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        margin: '24px 0',
                        gap: '16px'
                    }}>
                        <div style={{ flex: 1, height: '1px', background: 'rgba(0,0,0,0.1)' }} />
                        <span style={{ color: '#999', fontSize: '12px' }}>or continue with</span>
                        <div style={{ flex: 1, height: '1px', background: 'rgba(0,0,0,0.1)' }} />
                    </div>

                    {/* Social Buttons */}
                    <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={handleGoogleSignIn}
                            disabled={isSubmitting}
                            style={{
                                flex: 1,
                                padding: '14px',
                                borderRadius: '12px',
                                border: 'none',
                                background: '#DB4437',
                                color: 'white',
                                fontWeight: '600',
                                cursor: isSubmitting ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                fontSize: '14px',
                                opacity: isSubmitting ? 0.7 : 1
                            }}
                        >
                            <Mail size={18} />
                            Google
                        </motion.button>
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={handleFacebookSignIn}
                            disabled={isSubmitting}
                            style={{
                                flex: 1,
                                padding: '14px',
                                borderRadius: '12px',
                                border: 'none',
                                background: '#4267B2',
                                color: 'white',
                                fontWeight: '600',
                                cursor: isSubmitting ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                fontSize: '14px',
                                opacity: isSubmitting ? 0.7 : 1
                            }}
                        >
                            <span style={{ fontWeight: '800' }}>f</span>
                            Facebook
                        </motion.button>
                    </div>

                    {/* Guest Login */}
                    <motion.button
                        onClick={handleGuestMode}
                        whileHover={{ scale: 1.02, background: 'rgba(29, 185, 84, 0.1)' }}
                        whileTap={{ scale: 0.98 }}
                        disabled={isSubmitting}
                        style={{
                            width: '100%',
                            padding: '14px',
                            borderRadius: '12px',
                            border: '2px solid #1DB954',
                            background: 'transparent',
                            color: '#1DB954',
                            fontWeight: '600',
                            cursor: isSubmitting ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '8px',
                            opacity: isSubmitting ? 0.7 : 1
                        }}
                    >
                        <Sparkles size={18} />
                        Continue as Guest
                    </motion.button>
                </motion.div>

                {/* Terms */}
                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    style={{
                        marginTop: '24px',
                        fontSize: '12px',
                        color: '#666',
                        textAlign: 'center',
                        maxWidth: '300px'
                    }}
                >
                    By continuing, you agree to our{' '}
                    <Link href="/privacy" style={{ color: '#1DB954' }}>Terms of Service</Link>
                    {' '}and{' '}
                    <Link href="/privacy" style={{ color: '#1DB954' }}>Privacy Policy</Link>
                </motion.p>
            </div>

            <style>{`
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
}
