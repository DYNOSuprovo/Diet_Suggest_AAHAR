"use client";
import { useState } from 'react';
import { usePathname } from 'next/navigation';
import { MessageCircle } from 'lucide-react';
import FeedbackModal from './FeedbackModal';

export default function FeedbackButton() {
    const [isOpen, setIsOpen] = useState(false);
    const pathname = usePathname();

    // Only show on profile page
    if (pathname !== '/profile') return null;

    return (
        <>
            <button
                onClick={() => setIsOpen(true)}
                style={{
                    position: 'fixed',
                    bottom: '90px',
                    right: '20px',
                    width: '56px',
                    height: '56px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #1DB954 0%, #4CAF50 100%)',
                    border: 'none',
                    boxShadow: '0 4px 12px rgba(29, 185, 84, 0.4)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1000,
                    transition: 'transform 0.2s, box-shadow 0.2s'
                }}
                onMouseOver={(e) => {
                    e.currentTarget.style.transform = 'scale(1.1)';
                    e.currentTarget.style.boxShadow = '0 6px 16px rgba(29, 185, 84, 0.5)';
                }}
                onMouseOut={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(29, 185, 84, 0.4)';
                }}
            >
                <MessageCircle size={24} color="white" />
            </button>
            <FeedbackModal isOpen={isOpen} onClose={() => setIsOpen(false)} />
        </>
    );
}
