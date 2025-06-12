import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io'; // Import for SocketException
import 'dart:math' as math;
import 'package:google_fonts/google_fonts.dart'; // Import the Google Fonts package
import 'package:flutter_markdown/flutter_markdown.dart'; // Import for Markdown rendering

void main() {
  runApp(const DietAdvisorApp());
}

// Main App Widget - Sets up the theme and initial route for the Diet Advisor app
class DietAdvisorApp extends StatelessWidget {
  const DietAdvisorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Diet Advisor',
      theme: ThemeData(
        useMaterial3: true,
        // Sophisticated Color Scheme with a focus on fresh greens and soft neutrals
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6B8E23), // Olive Green - primary base
          primary: const Color(0xFF6B8E23), // Olive Green - primary elements
          onPrimary: Colors.white, // Text color on primary
          secondary: const Color(0xFF8BC34A), // Light Green - secondary accent
          onSecondary: Colors.white, // Text color on secondary
          tertiary: const Color(0xFFC8E6C9), // Very light green - subtle accent
          onTertiary: Colors.black87, // Text color on tertiary
          background: const Color(0xFFF9FBF6), // Off-white/very light green background
          onBackground: Colors.black87, // Text color on background
          surface: Colors.white, // Pure white for cards, bubbles, etc.
          onSurface: Colors.black87, // Text color on surfaces
          error: Colors.red[600], // Error color
          onError: Colors.white, // Text color on error
          brightness: Brightness.light,
        ),
        // Google Fonts for a professional and modern look
        textTheme: TextTheme(
          displayLarge: GoogleFonts.montserrat(fontSize: 57, fontWeight: FontWeight.bold, color: Colors.black87),
          displayMedium: GoogleFonts.montserrat(fontSize: 45, fontWeight: FontWeight.bold, color: Colors.black87),
          displaySmall: GoogleFonts.montserrat(fontSize: 36, fontWeight: FontWeight.bold, color: Colors.black87),
          headlineLarge: GoogleFonts.montserrat(fontSize: 32, fontWeight: FontWeight.bold, color: Colors.black87),
          headlineMedium: GoogleFonts.montserrat(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.black87),
          headlineSmall: GoogleFonts.montserrat(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.black87),
          titleLarge: GoogleFonts.lato(fontSize: 22, fontWeight: FontWeight.w600, color: Colors.black87),
          titleMedium: GoogleFonts.lato(fontSize: 18, fontWeight: FontWeight.w600, color: Colors.black87), // Slightly increased
          titleSmall: GoogleFonts.lato(fontSize: 16, fontWeight: FontWeight.w600, color: Colors.black87), // Slightly increased
          bodyLarge: GoogleFonts.inter(fontSize: 16, color: Colors.black87),
          bodyMedium: GoogleFonts.inter(fontSize: 14, color: Colors.black87),
          bodySmall: GoogleFonts.inter(fontSize: 12, color: Colors.black87),
          labelLarge: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.black87),
          labelMedium: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w500, color: Colors.black87),
          labelSmall: GoogleFonts.inter(fontSize: 11, fontWeight: FontWeight.w500, color: Colors.black87),
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: const Color(0xFF6B8E23), // Olive Green app bar
          foregroundColor: Colors.white, // White text/icons on app bar
          elevation: 8, // More prominent shadow for app bar
          centerTitle: true,
          titleTextStyle: GoogleFonts.montserrat( // Stronger font for app bar title
            fontSize: 22,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF8BC34A), // Light Green button background
            foregroundColor: Colors.white, // White button text
            padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 18), // Slightly larger
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30), // Pill-shaped button
            ),
            elevation: 8, // More pronounced shadow
            shadowColor: Colors.black.withOpacity(0.3), // Darker, more defined shadow
            textStyle: GoogleFonts.lato(fontSize: 18, fontWeight: FontWeight.bold), // Sophisticated font for button
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(15), // Slightly more rounded
            borderSide: BorderSide.none,
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(15),
            borderSide: BorderSide(color: const Color(0xFF6B8E23), width: 2), // Olive Green focused border
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(15),
            borderSide: BorderSide(color: Colors.grey.shade300, width: 1), // Softer grey border
          ),
          errorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(15),
            borderSide: BorderSide(color: Colors.red[600]!, width: 1),
          ),
          focusedErrorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(15),
            borderSide: BorderSide(color: Colors.red[600]!, width: 2),
          ),
          contentPadding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20), // More padding
        ),
        cardTheme: CardThemeData(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)), // More rounded cards
          elevation: 5, // Increased elevation
          color: Colors.white.withOpacity(0.98), // Almost opaque white for cards
          shadowColor: Colors.black.withOpacity(0.15), // Subtle card shadow
        ),
        splashFactory: InkSparkle.splashFactory, // Modern ripple effect
      ),
      home: const DietHomeScreen(), // Start with the Diet Home Screen
      debugShowCheckedModeBanner: false, // Hide debug banner
    );
  }
}

// Home Screen with Canvas Decorations and Interactive Buttons for Diet App
class DietHomeScreen extends StatefulWidget {
  const DietHomeScreen({super.key});

  @override
  State<DietHomeScreen> createState() => _DietHomeScreenState();
}

class _DietHomeScreenState extends State<DietHomeScreen> with TickerProviderStateMixin {
  late AnimationController _iconPulsateController; // For pulsating central icon
  late AnimationController _backgroundWaveController; // For subtle background waves
  late Animation<double> _scaleAnimation; // For the pulsating icon's scale
  late Animation<Color?> _shadowColorAnimation; // For the pulsating icon's shadow color

  @override
  void initState() {
    super.initState();

    // Initialize controller for central icon pulsation
    _iconPulsateController = AnimationController(
      duration: const Duration(seconds: 3), // Moderate pulsation speed
      vsync: this,
    )..repeat(reverse: true); // Repeat back and forth

    _scaleAnimation = Tween<double>(begin: 1.0, end: 1.1).animate(
      CurvedAnimation(
        parent: _iconPulsateController,
        curve: Curves.easeInOut, // Smoother pulsation
      ),
    );

    _shadowColorAnimation = ColorTween(
      begin: Colors.lightGreen.withOpacity(0.3),
      end: Colors.lightGreen.withOpacity(0.7),
    ).animate(
      CurvedAnimation(
        parent: _iconPulsateController,
        curve: Curves.easeInOut, // Smoother shadow pulsation
      ),
    );


    // Initialize controller for background wave animation
    _backgroundWaveController = AnimationController(
      duration: const Duration(seconds: 20), // Slower, more serene wave movement
      vsync: this,
    )..repeat(reverse: true); // Repeat indefinitely
  }

  @override
  void dispose() {
    _iconPulsateController.dispose();
    _backgroundWaveController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      body: Stack(
        children: [
          // Dynamic Background Painter with subtle wave effect
          AnimatedBuilder(
            animation: _backgroundWaveController,
            builder: (context, child) {
              return CustomPaint(
                painter: _DietBackgroundPainter(
                  animationValue: _backgroundWaveController.value,
                  color1: colors.background!,
                  color2: colors.tertiary.withOpacity(0.3), // Use tertiary for subtle wave
                ),
                size: Size.infinite,
              );
            },
          ),

          // UI Elements (Title, Subtitle, Button) - positioned safely
          SafeArea(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Spacer(flex: 2),
                  // Central Diet Icon with Pulsating Animation and Enhanced Shadow
                  AnimatedBuilder(
                    animation: _iconPulsateController,
                    builder: (context, child) {
                      return Transform.scale(
                        scale: _scaleAnimation.value,
                        child: Container(
                          padding: const EdgeInsets.all(35), // Slightly larger padding
                          decoration: BoxDecoration(
                            color: colors.surface.withOpacity(0.95), // Brighter white with less transparency
                            shape: BoxShape.circle,
                            boxShadow: [
                              BoxShadow(
                                color: _shadowColorAnimation.value!, // Animated shadow color
                                blurRadius: 40, // Increased blur
                                spreadRadius: 10, // Increased spread
                                offset: const Offset(0, 15), // More vertical shadow
                              ),
                            ],
                          ),
                          child: Icon(
                            Icons.local_dining_rounded, // Diet-related icon
                            size: 90, // Larger icon
                            color: colors.primary, // Primary olive green color
                          ),
                        ),
                      );
                    },
                  ),
                  const SizedBox(height: 40), // More space
                  // Main title
                  Text(
                    'Your Elite Diet Advisor', // More professional phrasing
                    style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                          color: colors.onBackground,
                          fontWeight: FontWeight.bold,
                        ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 20), // More space
                  // Subtitle
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32.0), // Wider padding
                    child: Text(
                      'Unlock optimal health with personalized insights, meal tracking, and expert guidance tailored for you.', // More sophisticated copy
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            color: colors.onBackground.withOpacity(0.7),
                          ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  const Spacer(flex: 3),
                  // Start Chatting Button
                  ElevatedButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        PageRouteBuilder(
                          pageBuilder: (context, animation, secondaryAnimation) => const DietChatScreen(),
                          transitionsBuilder: (context, animation, secondaryAnimation, child) {
                            const begin = Offset(0.0, 1.0); // Slide in from bottom
                            const end = Offset.zero;
                            const curve = Curves.easeOutCubic;
                            var tween = Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
                            return SlideTransition(
                              position: animation.drive(tween),
                              child: child,
                            );
                          },
                        ),
                      );
                    },
                    icon: const Icon(Icons.chat_bubble_outline_rounded),
                    label: const Text('Begin Your Journey'), // More inviting text
                    style: ElevatedButton.styleFrom(
                      backgroundColor: colors.primary, // Primary green button
                      foregroundColor: colors.onPrimary,
                      padding: const EdgeInsets.symmetric(horizontal: 45, vertical: 20), // Larger button
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(35), // More rounded, modern pill shape
                      ),
                      elevation: 12, // Higher elevation for prominence
                      shadowColor: colors.primary.withOpacity(0.6), // Stronger shadow
                      textStyle: GoogleFonts.lato(fontSize: 19, fontWeight: FontWeight.bold), // Larger, sophisticated font
                    ),
                  ),
                  const Spacer(flex: 1),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Custom Painter for the Diet Home Screen Background
class _DietBackgroundPainter extends CustomPainter {
  final double animationValue;
  final Color color1;
  final Color color2;

  _DietBackgroundPainter({required this.animationValue, required this.color1, required this.color2});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();
    final rect = Rect.fromLTWH(0, 0, size.width, size.height);

    // Subtle gradient background
    final gradient = LinearGradient(
      begin: Alignment.topLeft,
      end: Alignment.bottomRight,
      colors: [color1, color2],
    );
    paint.shader = gradient.createShader(rect);
    canvas.drawRect(rect, paint);

    // Add subtle, organic wave patterns with soft glow effect
    final wavePaint = Paint()
      ..color = color2.withOpacity(0.6) // Slightly more opaque than background
      ..style = PaintingStyle.fill
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, 15); // Soft blur/glow

    final path = Path();
    path.moveTo(0, size.height * 0.4 + math.sin(animationValue * 2 * math.pi) * 25); // Increased wave amplitude
    path.quadraticBezierTo(
      size.width * 0.3,
      size.height * 0.3 + math.cos(animationValue * 2 * math.pi) * 35, // Increased amplitude
      size.width * 0.6,
      size.height * 0.5 + math.sin(animationValue * 2 * math.pi) * 30, // Increased amplitude
    );
    path.quadraticBezierTo(
      size.width * 0.9,
      size.height * 0.6 + math.cos(animationValue * 2 * math.pi) * 25, // Increased amplitude
      size.width,
      size.height * 0.5 + math.sin(animationValue * 2 * math.pi) * 20, // Increased amplitude
    );
    path.lineTo(size.width, size.height);
    path.lineTo(0, size.height);
    path.close();
    canvas.drawPath(path, wavePaint);

    final path2 = Path();
    wavePaint.color = color2.withOpacity(0.4); // Less opaque for second wave
    wavePaint.maskFilter = MaskFilter.blur(BlurStyle.normal, 10); // Slightly less blur
    path2.moveTo(0, size.height * 0.7 + math.cos(animationValue * 2 * math.pi) * 30); // Increased amplitude
    path2.quadraticBezierTo(
      size.width * 0.4,
      size.height * 0.8 + math.sin(animationValue * 2 * math.pi) * 40, // Increased amplitude
      size.width * 0.8,
      size.height * 0.7 + math.cos(animationValue * 2 * math.pi) * 35, // Increased amplitude
    );
    path2.lineTo(size.width, size.height);
    path2.lineTo(0, size.height);
    path2.close();
    canvas.drawPath(path2, wavePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    if (oldDelegate is _DietBackgroundPainter) {
      return oldDelegate.animationValue != animationValue || oldDelegate.color1 != color1 || oldDelegate.color2 != color2;
    }
    return true;
  }
}

// Chat Screen with Diet Advisor Interface
class DietChatScreen extends StatefulWidget {
  const DietChatScreen({super.key});

  @override
  State<DietChatScreen> createState() => _DietChatScreenState();
}

class _DietChatScreenState extends State<DietChatScreen> with TickerProviderStateMixin {
  final _controller = TextEditingController();
  final List<Map<String, String>> _messages = [];
  final ScrollController _scrollController = ScrollController();
  final String _sessionId = DateTime.now().microsecondsSinceEpoch.toString(); // Unique session ID
  bool _isLoading = false;
  late AnimationController _loadingAnimationController;
  late AnimationController _backgroundAnimationController;
  late AnimationController _sendButtonAnimationController;

  final String _backendUrl = 'https://diet-suggest-aahar.onrender.com/chat'; // Your Render backend URL

  @override
  void initState() {
    super.initState();
    _loadingAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _backgroundAnimationController = AnimationController(
      duration: const Duration(seconds: 20), // Slower background animation
      vsync: this,
    )..repeat(reverse: true);

    _sendButtonAnimationController = AnimationController(
      duration: const Duration(milliseconds: 900), // Subtle pulsation
      vsync: this,
    )..repeat(reverse: true);

    // Initial greeting from the Diet Advisor
    _messages.insert(0, {
      "role": "assistant",
      "content": "Hello! I'm your personal Diet Advisor. Ask me anything about nutrition, recipes, or healthy living!"
    });
  }

  @override
  void dispose() {
    _loadingAnimationController.dispose();
    _backgroundAnimationController.dispose();
    _sendButtonAnimationController.dispose();
    _scrollController.dispose();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _isLoading) return;

    setState(() {
      _messages.insert(0, {"role": "user", "content": text});
      _isLoading = true;
    });
    _controller.clear();
    _loadingAnimationController.forward(from: 0.0);
    _scrollToBottom(); // Scroll to bottom when user sends message

    try {
      final response = await http.post(
        Uri.parse(_backendUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "query": text,
          "session_id": _sessionId,
        }),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(utf8.decode(response.bodyBytes));
        final String advisorResponse = responseData['answer'] ?? 'Sorry, I could not get a response.';
        setState(() {
          _messages.insert(0, {"role": "assistant", "content": advisorResponse});
        });
      } else {
        setState(() {
          _messages.insert(0, {
            "role": "assistant",
            "content": "Error connecting to advisor (Code: ${response.statusCode}). Please try again."
          });
        });
        print('Backend error: ${response.statusCode} - ${response.body}');
      }
    } on SocketException catch (e) {
      setState(() {
        _messages.insert(0, {
          "role": "assistant",
          "content": "Network error: Could not reach the Diet Advisor. Please check your internet connection. ($e)",
        });
      });
      print('SocketException: $e');
    } on http.ClientException catch (e) {
      setState(() {
        _messages.insert(0, {
          "role": "assistant",
          "content": "HTTP error: Something went wrong with the request. ($e)",
        });
      });
      print('HttpClientException: $e');
    } catch (e) {
      setState(() {
        _messages.insert(0, {
          "role": "assistant",
          "content": "An unexpected error occurred: $e. Please try again.",
        });
      });
      print('Unexpected error: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
      _loadingAnimationController.reverse();
      _scrollToBottom(); // Scroll to bottom after response
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        // Ensuring messages stay in view by adjusting jumpTo logic
        // This will attempt to keep the most recent message at the very bottom
        // or as close to it as possible.
        _scrollController.jumpTo(_scrollController.position.minScrollExtent); // Ensures it scrolls to the very bottom
      }
    });
  }

  // Widget to display a loading message
  Widget _buildLoadingMessage() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 16, right: 60),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface.withOpacity(0.95), // White loading bubble
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(20),
            topRight: Radius.circular(20),
            bottomRight: Radius.circular(20),
            bottomLeft: Radius.circular(8), // Subtle tail
          ),
          boxShadow: [
            BoxShadow(
              color: Theme.of(context).colorScheme.primary.withOpacity(0.2), // Green shadow
              blurRadius: 15,
              spreadRadius: 3,
              offset: const Offset(0, 8),
            ),
          ],
          border: Border.all(color: Theme.of(context).colorScheme.primary.withOpacity(0.5), width: 1.0),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(
                strokeWidth: 2.5,
                valueColor: AlwaysStoppedAnimation<Color>(
                  Theme.of(context).colorScheme.primary, // Green loading indicator
                ),
              ),
            ),
            const SizedBox(width: 12),
            Text(
              'Getting advice...',
              style: TextStyle(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
                fontStyle: FontStyle.italic,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Widget to build individual chat message bubbles
  Widget _buildMessageBubble(Map<String, String> msg, bool isUser) {
    return _AnimatedChatMessageBubble(
      message: msg["content"]!,
      isUser: isUser,
      color: isUser
          ? Theme.of(context).colorScheme.primary // Primary for user
          : Theme.of(context).colorScheme.surface, // Surface for assistant
      textColor: isUser
          ? Theme.of(context).colorScheme.onPrimary // OnPrimary for user
          : Theme.of(context).colorScheme.onSurface, // OnSurface for assistant
      assistantAvatarColor: Theme.of(context).colorScheme.secondary, // Secondary for assistant avatar
      assistantIconColor: Theme.of(context).colorScheme.onSecondary,
      userAvatarColor: Theme.of(context).colorScheme.primary, // Primary for user avatar
      userIconColor: Theme.of(context).colorScheme.onPrimary,
      bubbleShadowColor: isUser
          ? Theme.of(context).colorScheme.primary.withOpacity(0.2) // Subtle shadow for user
          : Colors.black.withOpacity(0.1), // Subtle shadow for assistant
    );
  }

  // Widget to build the input area at the bottom of the chat screen
  Widget _buildInputArea() {
    final colors = Theme.of(context).colorScheme;
    return Material(
      elevation: 15, // Prominent floating effect
      shadowColor: Colors.black.withOpacity(0.2), // Darker shadow
      borderRadius: const BorderRadius.vertical(top: Radius.circular(25)), // Rounded top corners
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12), // Standard padding
        decoration: BoxDecoration(
          color: colors.surface, // White background
          borderRadius: const BorderRadius.vertical(top: Radius.circular(25)),
          border: Border.all(color: colors.tertiary, width: 0.5), // Subtle border
        ),
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _controller,
                onSubmitted: (_) => _sendMessage(),
                style: GoogleFonts.inter(fontSize: 15, color: colors.onSurface), // Input text style
                decoration: InputDecoration(
                  hintText: 'Ask about recipes, nutrition, or health...',
                  hintStyle: GoogleFonts.inter(color: colors.onSurface.withOpacity(0.5), fontSize: 15), // Hint text style
                  prefixIcon: Icon(Icons.food_bank_outlined, color: colors.secondary), // Light green icon
                  filled: true,
                  fillColor: colors.background, // Light green background for text field
                  contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(30),
                    borderSide: BorderSide.none,
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(30),
                    borderSide: BorderSide(color: colors.tertiary, width: 1), // Subtle border
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(30),
                    borderSide: BorderSide(color: colors.primary, width: 2), // Primary green on focus
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            AnimatedBuilder(
              animation: _sendButtonAnimationController,
              builder: (context, child) {
                return Transform.scale(
                  scale: 1.0 + (math.sin(_sendButtonAnimationController.value * math.pi) * 0.03), // Subtle pulsation
                  child: Container(
                    decoration: BoxDecoration(
                      color: _isLoading ? colors.onSurface.withOpacity(0.4) : colors.primary, // Grey when loading, primary green when active
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: colors.primary.withOpacity(0.4),
                          blurRadius: 10,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                    child: IconButton(
                      onPressed: _isLoading ? null : _sendMessage, // Disable when loading
                      icon: _isLoading ? const Icon(Icons.hourglass_empty_rounded) : const Icon(Icons.send_rounded), // Different icon for loading
                      color: Colors.white,
                      tooltip: 'Send',
                      padding: const EdgeInsets.all(16),
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background painter for the chat area
          AnimatedBuilder(
            animation: _backgroundAnimationController,
            builder: (context, child) {
              return CustomPaint(
                painter: _ChatBackgroundPainter(
                  colors: Theme.of(context).colorScheme,
                  animationValue: _backgroundAnimationController.value,
                ),
                size: Size.infinite,
              );
            },
          ),
          Column(
            children: [
              AppBar(
                elevation: 4,
                backgroundColor: Colors.transparent, // Make app bar transparent
                flexibleSpace: Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Theme.of(context).colorScheme.primary, Theme.of(context).colorScheme.secondary], // Gradient app bar
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.15), // Stronger app bar shadow
                        blurRadius: 10,
                        offset: const Offset(0, 5),
                      ),
                    ],
                  ),
                ),
                iconTheme: const IconThemeData(color: Colors.white),
                title: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8), // Larger padding for icon background
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.25), // More visible icon background
                        borderRadius: BorderRadius.circular(12), // More rounded
                      ),
                      child: Icon(Icons.local_dining_rounded, size: 24, color: Colors.white), // Larger icon
                    ),
                    const SizedBox(width: 12), // More space
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            'Diet Advisor Chat',
                            style: GoogleFonts.montserrat( // Stronger font
                              color: Colors.white,
                              fontSize: 19, // Slightly larger
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            'Your intelligent guide to healthy eating', // More sophisticated subtitle
                            style: GoogleFonts.lato( // Subtitle font
                              color: Colors.white70,
                              fontSize: 13, // Slightly larger
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              Expanded(
                child: ListView.builder(
                  controller: _scrollController,
                  reverse: true, // Keep reverse: true for chat-like behavior
                  padding: const EdgeInsets.all(16),
                  itemCount: _messages.length + (_isLoading ? 1 : 0),
                  itemBuilder: (context, index) {
                    if (_isLoading && index == 0) return _buildLoadingMessage();
                    final msgIndex = _isLoading ? index - 1 : index;
                    final msg = _messages[msgIndex];
                    final isUser = msg["role"] == "user";
                    // Now pass the animation properties to the new AnimatedChatMessageBubble
                    return _buildMessageBubble(msg, isUser);
                  },
                ),
              ),
              _buildInputArea(),
            ],
          ),
        ],
      ),
    );
  }
}

// Custom Painter for the Chat Screen Background (subtle patterns)
class _ChatBackgroundPainter extends CustomPainter {
  final ColorScheme colors;
  final double animationValue;

  _ChatBackgroundPainter({required this.colors, required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();
    final rect = Rect.fromLTWH(0, 0, size.width, size.height);

    // Softer gradient background for chat
    final gradient = LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [colors.tertiary!.withOpacity(0.4), colors.background!], // Softer gradient
      stops: const [0.0, 1.0],
    );
    paint.shader = gradient.createShader(rect);
    canvas.drawRect(rect, paint);

    // Add very subtle, light, blurred patterns for sophisticated texture
    final patternPaint = Paint()
      ..color = colors.primary.withOpacity(0.02) // Extremely light green for patterns
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.6
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, 2); // Very subtle blur

    // Example: Wavy lines that slowly shift
    for (double i = -size.width * 0.5; i < size.width * 1.5; i += 80) { // Wider range for lines
      final path = Path();
      path.moveTo(i + animationValue * 50, size.height * 0.1);
      path.quadraticBezierTo(
        i + animationValue * 50 + 40,
        size.height * 0.3 + math.sin(animationValue * math.pi * 2 + i * 0.01) * 10,
        i + animationValue * 50 + 80,
        size.height * 0.5 + math.cos(animationValue * math.pi * 2 + i * 0.01) * 10,
      );
      path.quadraticBezierTo(
        i + animationValue * 50 + 120,
        size.height * 0.7 + math.sin(animationValue * math.pi * 2 + i * 0.01) * 10,
        i + animationValue * 50 + 160,
        size.height * 0.9,
      );
      canvas.drawPath(path, patternPaint);
    }

    // Example: Small, shimmering dots
    final random = math.Random(42); // Fixed seed for consistent appearance
    for (int i = 0; i < 70; i++) { // More dots
      final x = random.nextDouble() * size.width;
      final y = random.nextDouble() * size.height;
      final radius = random.nextDouble() * 3 + 1; // Smaller radius
      final opacity = (math.sin(animationValue * math.pi * 4 + (x + y) * 0.005) + 1) / 2; // Faster shimmer
      patternPaint.color = colors.secondary.withOpacity(0.03 + opacity * 0.03); // Very subtle shimmering teal
      canvas.drawCircle(Offset(x, y), radius, patternPaint..style = PaintingStyle.fill);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    if (oldDelegate is _ChatBackgroundPainter) {
      return oldDelegate.animationValue != animationValue || oldDelegate.colors != colors;
    }
    return true;
  }
}


// New StatefulWidget for individual chat message animations
class _AnimatedChatMessageBubble extends StatefulWidget {
  final String message;
  final bool isUser;
  final Color color;
  final Color textColor;
  final Color assistantAvatarColor;
  final Color assistantIconColor;
  final Color userAvatarColor;
  final Color userIconColor;
  final Color bubbleShadowColor;


  const _AnimatedChatMessageBubble({
    required this.message,
    required this.isUser,
    required this.color,
    required this.textColor,
    required this.assistantAvatarColor,
    required this.assistantIconColor,
    required this.userAvatarColor,
    required this.userIconColor,
    required this.bubbleShadowColor,
    super.key,
  });

  @override
  State<_AnimatedChatMessageBubble> createState() => _AnimatedChatMessageBubbleState();
}

class _AnimatedChatMessageBubbleState extends State<_AnimatedChatMessageBubble> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300), // Quick entrance animation
    );

    _slideAnimation = Tween<Offset>(
      begin: Offset(widget.isUser ? 0.5 : -0.5, 0.0), // Slide from left/right
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutCubic,
    ));

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: Curves.easeIn,
      ),
    );

    _animationController.forward(); // Start animation when the widget is built
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: SlideTransition(
        position: _slideAnimation,
        child: Align(
          alignment: widget.isUser ? Alignment.centerRight : Alignment.centerLeft,
          child: ConstrainedBox(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width * 0.75,
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (!widget.isUser) // Assistant avatar
                  Padding(
                    padding: const EdgeInsets.only(right: 8.0, bottom: 4.0),
                    child: CircleAvatar(
                      radius: 16,
                      backgroundColor: widget.assistantAvatarColor,
                      child: Icon(Icons.fitness_center_rounded, size: 18, color: widget.assistantIconColor),
                    ),
                  ),
                Flexible(
                  child: Container(
                    margin: const EdgeInsets.only(bottom: 12),
                    padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
                    decoration: BoxDecoration(
                      color: widget.color,
                      borderRadius: BorderRadius.only(
                        topLeft: const Radius.circular(16),
                        topRight: const Radius.circular(16),
                        bottomLeft: Radius.circular(widget.isUser ? 16 : 4),
                        bottomRight: Radius.circular(widget.isUser ? 4 : 16),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: widget.bubbleShadowColor,
                          blurRadius: 10,
                          spreadRadius: 1,
                          offset: const Offset(0, 5),
                        ),
                      ],
                      border: Border.all(
                        color: widget.isUser
                            ? Theme.of(context).colorScheme.primary.withOpacity(0.8)
                            : Colors.grey.shade300,
                        width: 1.0,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Use MarkdownBody to render the message content
                        MarkdownBody(
                          data: widget.message,
                          selectable: true, // Allow text selection
                          styleSheet: MarkdownStyleSheet(
                            p: TextStyle(
                              color: widget.textColor,
                              fontSize: 14,
                              height: 1.3,
                              fontFamily: GoogleFonts.inter().fontFamily, // Apply Inter font
                            ),
                            // Make single asterisks render as bold by styling `em` (emphasis)
                            em: GoogleFonts.inter( // Italic text style (default)
                              fontWeight: FontWeight.bold, // Make single asterisks bold
                              fontStyle: FontStyle.normal, // Override default italic style
                              color: widget.textColor,
                              fontSize: 14,
                            ),
                            strong: GoogleFonts.inter( // Bold text style (for **...**)
                              fontWeight: FontWeight.bold,
                              color: widget.textColor,
                              fontSize: 14,
                            ),
                            tableBody: GoogleFonts.inter( // Table text style
                              color: widget.textColor,
                              fontSize: 14,
                            ),
                            tableHead: GoogleFonts.inter( // Table header style
                              fontWeight: FontWeight.bold,
                              color: widget.textColor,
                              fontSize: 14,
                            ),
                            tableBorder: TableBorder.all( // Table border
                              color: Theme.of(context).colorScheme.onSurface.withOpacity(0.2),
                              width: 1.0,
                            ),
                            // Add other Markdown styles as needed (e.g., for headers, lists)
                            h1: GoogleFonts.montserrat(fontSize: 24, fontWeight: FontWeight.bold, color: widget.textColor),
                            h2: GoogleFonts.montserrat(fontSize: 22, fontWeight: FontWeight.bold, color: widget.textColor),
                            h3: GoogleFonts.montserrat(fontSize: 20, fontWeight: FontWeight.bold, color: widget.textColor),
                            // Removed listItem as it's not supported in flutter_markdown 0.7.7+1
                          ),
                          // Override link and image builders if custom rendering is needed
                          onTapLink: (text, href, title) {
                            // Handle link taps if necessary
                          },
                        ),
                      ],
                    ),
                  ),
                ),
                if (widget.isUser) // User avatar
                  Padding(
                    padding: const EdgeInsets.only(left: 8.0, bottom: 4.0),
                    child: CircleAvatar(
                      radius: 16,
                      backgroundColor: widget.userAvatarColor,
                      child: Icon(Icons.person_rounded, size: 18, color: widget.userIconColor),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
