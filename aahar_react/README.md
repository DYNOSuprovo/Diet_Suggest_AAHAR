# Aahar - Your Personal Wellness Companion 🥗💧

Aahar is a comprehensive diet and wellness tracking application designed to help users achieve their health goals through smart tracking and personalized insights. Built with modern web technologies and optimized for mobile devices, Aahar provides a seamless experience for monitoring nutrition, hydration, and weight progress.

## 🌟 Key Features

### 1. **Smart Onboarding & Personalization**
*   **Personalized Goals:** Calculates daily calorie and water intake goals based on your age, gender, weight, height, and activity level.
*   **BMI Calculation:** Automatically computes your Body Mass Index (BMI) to guide your health journey.
*   **Dietary Preferences:** Set preferences like Vegetarian, Gluten-Free, Low-Carb, etc.

### 2. **Hydration Tracker 💧**
*   **Visual Progress:** A beautiful circular progress ring shows your daily hydration status at a glance.
*   **Quick Add:** One-tap buttons for common amounts (250ml, 500ml, etc.) and custom entry options.
*   **Weekly Analysis:** Interactive charts display your water intake history over the last 7 days.
*   **Smart Resets:** Automatically resets your daily log at midnight while preserving your history.

### 3. **Meal Management 🍎**
*   **Daily Logs:** Track Breakfast, Lunch, Snacks, and Dinner separately.
*   **Calorie Tracking:** Monitor your daily calorie intake against your personalized goal.
*   **History:** Keep a record of your meals to identify eating patterns.

### 4. **Weight & Progress Tracking 📈**
*   **Weight Trend Chart:** Visualize your weight changes over time with a dynamic line graph.
*   **Historical Data:** The app intelligently stores your weight history to show real progress trends.

### 5. **AI-Powered Health Assistant 🤖**
*   Integrated AI chat interface (powered by Gemini/Groq) to answer your diet and nutrition questions instantly.

### 6. **Privacy-First & Offline Capable 🔒**
*   **Local Storage:** All your data is stored locally on your device. No external servers means your data stays with you.
*   **Offline Access:** The app works perfectly without an internet connection (except for AI features).

---

## 🛠️ Tech Stack

*   **Framework:** [Next.js 14](https://nextjs.org/) (App Router)
*   **UI Library:** [React](https://react.dev/)
*   **Styling:** CSS Modules / Global CSS
*   **Mobile Runtime:** [Capacitor](https://capacitorjs.com/) (for Android)
*   **Charts:** [Chart.js](https://www.chartjs.org/) & [React-Chartjs-2](https://react-chartjs-2.js.org/)
*   **Icons:** [Lucide React](https://lucide.dev/)
*   **AI Integration:** Google Gemini / Groq API

---

## 🚀 Getting Started

### Prerequisites
*   Node.js (v18 or higher)
*   npm or yarn
*   Android Studio (for mobile build)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DYNOSuprovo/aahar_react.git
    cd aahar_react
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Environment Setup:**
    Create a `.env.local` file in the root directory and add your API keys:
    ```env
    NEXT_PUBLIC_GEMINI_API_KEY="your_gemini_key"
    GROQ_API_KEY="your_groq_key"
    ```

4.  **Run Locally:**
    ```bash
    npm run dev
    ```
    Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📱 Building for Android

Aahar is configured with Capacitor to run as a native Android app.

1.  **Build the Next.js project:**
    ```bash
    npm run build
    ```

2.  **Sync with Capacitor:**
    ```bash
    npx cap sync android
    ```

3.  **Open in Android Studio:**
    ```bash
    npx cap open android
    ```
    From here, you can run the app on an emulator or a connected physical device.

---

## 📂 Project Structure

```
aahar/
├── android/              # Android native project files
├── public/               # Static assets (images, icons)
├── src/
│   ├── app/              # Next.js App Router pages
│   │   ├── chat/         # AI Chat page
│   │   ├── dashboard/    # Main dashboard
│   │   ├── onboarding/   # User setup flow
│   │   ├── profile/      # User profile & settings
│   │   ├── water/        # Water tracker
│   │   └── page.js       # Home/Landing page
│   ├── components/       # Reusable UI components (BottomNav, etc.)
│   ├── context/          # Global State (UserContext.js)
│   └── lib/              # Utilities and configuration
├── .env.local            # Environment variables (not committed)
├── capacitor.config.json # Capacitor configuration
└── next.config.js        # Next.js configuration
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [Suprovo](https://github.com/DYNOSuprovo)
