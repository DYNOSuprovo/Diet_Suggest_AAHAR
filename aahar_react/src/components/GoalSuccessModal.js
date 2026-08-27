"use client";
import { useEffect } from 'react';
import { X, Trophy } from 'lucide-react';

export default function GoalSuccessModal({ isOpen, onClose }) {
    if (!isOpen) return null;

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '20px',
            animation: 'fadeIn 0.3s ease-out'
        }}>
            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes scaleUp {
                    from { transform: scale(0.8); opacity: 0; }
                    to { transform: scale(1); opacity: 1; }
                }
            `}</style>
            <div style={{
                background: 'white',
                borderRadius: '24px',
                padding: '40px 32px',
                maxWidth: '360px',
                width: '100%',
                position: 'relative',
                boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
                textAlign: 'center',
                animation: 'scaleUp 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
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
                        padding: '8px',
                        color: '#9E9E9E'
                    }}
                >
                    <X size={24} />
                </button>

                {/* Trophy/Icon */}
                <div style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    background: '#E8F5E9',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 24px auto'
                }}>
                    <Trophy size={40} color="#1DB954" />
                </div>

                {/* Title */}
                <h2 style={{
                    fontSize: '24px',
                    fontWeight: '800',
                    color: '#1DB954',
                    marginBottom: '12px'
                }}>
                    Congratulations!
                </h2>

                {/* Message */}
                <p style={{
                    fontSize: '16px',
                    color: '#616161',
                    marginBottom: '32px',
                    lineHeight: '1.5'
                }}>
                    You've reached your daily water goal. Great job staying hydrated! 💧
                </p>

                {/* Action Button */}
                <button
                    onClick={onClose}
                    style={{
                        width: '100%',
                        padding: '16px',
                        background: 'linear-gradient(135deg, #42A5F5 0%, #2962FF 100%)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '16px',
                        fontSize: '16px',
                        fontWeight: '700',
                        cursor: 'pointer',
                        boxShadow: '0 4px 12px rgba(41, 98, 255, 0.2)',
                        transition: 'transform 0.2s'
                    }}
                    onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
                    onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                >
                    Awesome!
                </button>
            </div>
        </div>
    );
}
