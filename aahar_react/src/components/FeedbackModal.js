"use client";
import { useState } from 'react';
import { X, Star, Send, Loader2 } from 'lucide-react';
import { saveFeedback } from '../lib/firebase';
import { useAuth } from '../context/AuthContext';
import { useUser } from '../context/UserContext';

export default function FeedbackModal({ isOpen, onClose }) {
    const [rating, setRating] = useState(0);
    const [hoverRating, setHoverRating] = useState(0);
    const [feedback, setFeedback] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Get user info - hooks must be called unconditionally
    const { currentUser } = useAuth();
    const { user } = useUser();

    if (!isOpen) return null;

    const handleSubmit = async () => {
        if (isSubmitting) return;

        setIsSubmitting(true);

        // Simple feedback data - only name/id and feedback
        const feedbackData = {
            rating,
            comment: feedback.trim(),
            userName: user?.name || currentUser?.displayName || 'Anonymous',
            userEmail: user?.email || currentUser?.email || null
        };

        // Save to Realtime Database
        const result = await saveFeedback(currentUser?.uid || null, feedbackData);

        if (result.success) {
            console.log('Feedback saved');
        }

        setIsSubmitting(false);
        setSubmitted(true);
        setTimeout(() => {
            onClose();
            setSubmitted(false);
            setRating(0);
            setFeedback('');
        }, 2000);
    };

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '20px'
        }}>
            <div style={{
                background: 'white',
                borderRadius: '24px',
                padding: '32px',
                maxWidth: '400px',
                width: '100%',
                position: 'relative',
                boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
            }}>
                {/* Close button */}
                <button
                    onClick={onClose}
                    style={{
                        position: 'absolute',
                        top: '16px',
                        right: '16px',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '8px'
                    }}
                >
                    <X size={24} color="#757575" />
                </button>

                {!submitted ? (
                    <>
                        {/* Title */}
                        <h2 style={{
                            fontSize: '24px',
                            fontWeight: '700',
                            marginBottom: '8px',
                            color: '#1A1A1A'
                        }}>
                            How's your experience?
                        </h2>
                        <p style={{
                            fontSize: '14px',
                            color: '#757575',
                            marginBottom: '24px'
                        }}>
                            Your feedback helps us improve Aahar
                        </p>

                        {/* Star Rating */}
                        <div style={{
                            display: 'flex',
                            gap: '8px',
                            justifyContent: 'center',
                            marginBottom: '24px'
                        }}>
                            {[1, 2, 3, 4, 5].map((star) => (
                                <button
                                    key={star}
                                    onClick={() => setRating(star)}
                                    onMouseEnter={() => setHoverRating(star)}
                                    onMouseLeave={() => setHoverRating(0)}
                                    style={{
                                        background: 'none',
                                        border: 'none',
                                        cursor: 'pointer',
                                        padding: '4px',
                                        transition: 'transform 0.2s'
                                    }}
                                >
                                    <Star
                                        size={32}
                                        fill={star <= (hoverRating || rating) ? '#FFD700' : 'none'}
                                        color={star <= (hoverRating || rating) ? '#FFD700' : '#E0E0E0'}
                                    />
                                </button>
                            ))}
                        </div>

                        {/* Feedback Text */}
                        <textarea
                            value={feedback}
                            onChange={(e) => setFeedback(e.target.value)}
                            placeholder="Tell us more... (optional)"
                            style={{
                                width: '100%',
                                minHeight: '100px',
                                padding: '12px',
                                border: '1px solid #E0E0E0',
                                borderRadius: '12px',
                                fontSize: '14px',
                                fontFamily: 'inherit',
                                resize: 'vertical',
                                marginBottom: '20px'
                            }}
                        />

                        {/* Submit Button */}
                        <button
                            onClick={handleSubmit}
                            disabled={rating === 0 || isSubmitting}
                            style={{
                                width: '100%',
                                padding: '14px',
                                background: rating > 0 ? '#1DB954' : '#E0E0E0',
                                color: 'white',
                                border: 'none',
                                borderRadius: '12px',
                                fontSize: '16px',
                                fontWeight: '600',
                                cursor: rating > 0 && !isSubmitting ? 'pointer' : 'not-allowed',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                opacity: isSubmitting ? 0.8 : 1
                            }}
                        >
                            {isSubmitting ? (
                                <>
                                    <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                                    Sending...
                                </>
                            ) : (
                                <>
                                    <Send size={18} />
                                    Submit Feedback
                                </>
                            )}
                        </button>
                    </>
                ) : (
                    <div style={{ textAlign: 'center', padding: '20px 0' }}>
                        <div style={{ fontSize: '48px', marginBottom: '16px' }}>🎉</div>
                        <h3 style={{
                            fontSize: '20px',
                            fontWeight: '600',
                            color: '#1DB954',
                            marginBottom: '8px'
                        }}>
                            Thank you!
                        </h3>
                        <p style={{ fontSize: '14px', color: '#757575' }}>
                            Your feedback has been received
                        </p>
                    </div>
                )}
            </div>

            <style>{`
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
}
