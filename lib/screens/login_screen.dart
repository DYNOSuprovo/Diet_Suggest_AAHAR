
// lib/screens/login_screen.dart (Modified to add Sign Up navigation)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool _isPasswordVisible = false;
  int _selectedIndex = 0; // To manage Log In / Sign Up tabs

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        title: const Text(''),
        backgroundColor: Colors.transparent,
        foregroundColor: AppConstants.textColor,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Image.network(
                'https://placehold.co/80x80/E0E0E0/4CAF50?text=Logo', // Placeholder for app logo
                height: 80,
                width: 80,
              ),
            ),
            const SizedBox(height: 20),
            Text(
              'Welcome to\nHealthTrack',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: AppConstants.textColor,
                  ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 10),
            Text(
              'Your journey to a healthier you starts here.',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: AppConstants.greyColor,
                  ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 30),
            Container(
              decoration: BoxDecoration(
                color: AppConstants.lightGrey,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: GestureDetector(
                      onTap: () {
                        setState(() {
                          _selectedIndex = 0;
                        });
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        decoration: BoxDecoration(
                          color: _selectedIndex == 0 ? AppConstants.primaryColor : Colors.transparent,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Center(
                          child: Text(
                            'Log In',
                            style: TextStyle(
                              color: _selectedIndex == 0 ? Colors.white : AppConstants.textColor,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                  Expanded(
                    child: GestureDetector(
                      onTap: () {
                        setState(() {
                          _selectedIndex = 1;
                        });
                        Navigator.pushReplacementNamed(context, '/signup'); // Navigate to signup
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        decoration: BoxDecoration(
                          color: _selectedIndex == 1 ? AppConstants.primaryColor : Colors.transparent,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Center(
                          child: Text(
                            'Sign Up',
                            style: TextStyle(
                              color: _selectedIndex == 1 ? Colors.white : AppConstants.textColor,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
            if (_selectedIndex == 0) ...[
              Text(
                'Log In to Your Account',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 20),
              TextFormField(
                controller: _emailController,
                keyboardType: TextInputType.emailAddress,
                decoration: InputDecoration(
                  labelText: 'Email Address',
                  hintText: 'Enter email',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12.0),
                    borderSide: BorderSide.none,
                  ),
                  filled: true,
                  fillColor: AppConstants.lightGrey,
                  contentPadding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 20.0),
                ),
              ),
              const SizedBox(height: 20),
              TextFormField(
                controller: _passwordController,
                obscureText: !_isPasswordVisible,
                decoration: InputDecoration(
                  labelText: 'Password',
                  hintText: 'Enter password',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12.0),
                    borderSide: BorderSide.none,
                  ),
                  filled: true,
                  fillColor: AppConstants.lightGrey,
                  contentPadding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 20.0),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _isPasswordVisible ? Icons.visibility : Icons.visibility_off,
                      color: AppConstants.greyColor,
                    ),
                    onPressed: () {
                      setState(() {
                        _isPasswordVisible = !_isPasswordVisible;
                      });
                    },
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () {
                    // TODO: Implement forgot password logic
                  },
                  child: const Text(
                    'Forgot Password ?',
                    style: TextStyle(color: AppConstants.primaryColor, fontWeight: FontWeight.w600),
                  ),
                ),
              ),
              const SizedBox(height: 30),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    // TODO: Implement sign-in logic
                    Navigator.pushReplacementNamed(context, '/home');
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppConstants.primaryColor,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16.0),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.0),
                    ),
                  ),
                  child: const Text(
                    'Login',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
              const SizedBox(height: 30),
              const Center(
                child: Text(
                  'or',
                  style: TextStyle(
                    color: AppConstants.greyColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(height: 20),
              _SocialLoginButton(
                text: 'Continue with Email',
                icon: const Icon(Icons.email_outlined, color: AppConstants.textColor),
                onPressed: () {
                  // TODO: Implement email login
                },
              ),
              const SizedBox(height: 15),
              _SocialLoginButton(
                text: 'Continue with Apple',
                icon: const Icon(Icons.apple, color: AppConstants.textColor),
                onPressed: () {
                  // TODO: Implement Apple login
                },
              ),
              const SizedBox(height: 15),
              _SocialLoginButton(
                text: 'Continue with Youtube', // As per image, might be a typo for Google
                icon: Image.network(
                  'https://img.icons8.com/color/48/000000/youtube-play.png', // Placeholder
                  height: 24,
                  width: 24,
                ),
                onPressed: () {
                  // TODO: Implement YouTube/Google login
                },
              ),
              const SizedBox(height: 15),
              _SocialLoginButton(
                text: 'Continue with Facebook',
                icon: Image.network(
                  'https://img.icons8.com/color/48/000000/facebook-new.png', // Placeholder
                  height: 24,
                  width: 24,
                ),
                onPressed: () {
                  // TODO: Implement Facebook login
                },
              ),
            ]
          ],
        ),
      ),
    );
  }
}

class _SocialLoginButton extends StatelessWidget {
  final String text;
  final Widget icon;
  final VoidCallback onPressed;

  const _SocialLoginButton({
    required this.text,
    required this.icon,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16.0),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12.0),
          ),
          side: const BorderSide(color: AppConstants.lightGrey),
          backgroundColor: AppConstants.lightGrey,
          foregroundColor: AppConstants.textColor,
        ),
        icon: icon,
        label: Text(
          text,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
        ),
      ),
    );
  }
}
