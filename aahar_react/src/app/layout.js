import './globals.css';
import BottomNav from '../components/BottomNav';
import FeedbackButton from '../components/FeedbackButton';
import BackButtonHandler from '../components/BackButtonHandler';
import { UserProvider } from '../context/UserContext';
import { AuthProvider } from '../context/AuthContext';
import { ThemeProvider } from '../context/ThemeContext';
import { LanguageProvider } from '../context/LanguageContext';

export const metadata = {
    title: 'Aahar - Nutrition with Tradition',
    description: 'Track your nutrition with traditional Indian foods. Smart calorie tracking, AI assistant, and personalized meal planning.',
    manifest: '/manifest.json',
    themeColor: '#1DB954',
    viewport: {
        width: 'device-width',
        initialScale: 1,
        maximumScale: 1,
        userScalable: false,
        viewportFit: 'cover',
    },
    icons: {
        icon: '/icon.svg',
        apple: '/icon.svg',
    },
};

export default function RootLayout({ children }) {
    const GA_ID = process.env.NEXT_PUBLIC_GA_ID;

    return (
        <html lang="en" suppressHydrationWarning>
            <head>
                {/* Google Analytics - Only loads if GA_ID is set */}
                {GA_ID && (
                    <>
                        <script async src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}></script>
                        <script
                            dangerouslySetInnerHTML={{
                                __html: `
                                    window.dataLayer = window.dataLayer || [];
                                    function gtag(){dataLayer.push(arguments);}
                                    gtag('js', new Date());
                                    gtag('config', '${GA_ID}', {
                                        page_path: window.location.pathname,
                                        anonymize_ip: true,
                                    });
                                `,
                            }}
                        />
                    </>
                )}
            </head>
            <body>
                <ThemeProvider>
                    <LanguageProvider>
                        <div className="container">
                            <AuthProvider>
                                <UserProvider>
                                    {children}
                                </UserProvider>
                            </AuthProvider>
                            <BackButtonHandler />
                            <BottomNav />
                            <FeedbackButton />
                        </div>
                    </LanguageProvider>
                </ThemeProvider>
            </body>
        </html>
    );
}
