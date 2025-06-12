
// lib/utils/app_constants.dart (No change from previous version, kept for completeness)
import 'package:flutter/material.dart';

class AppConstants {
  static const Color primaryColor = Color(0xFF4CAF50); // A shade of green
  static const Color accentColor = Color(0xFF8BC34A); // A lighter green
  static const Color textColor = Colors.black87;
  static const Color lightGrey = Color(0xFFF5F5F5);
  static const Color greyColor = Color(0xFFB0B0B0);
  static const Color darkBlue = Color(0xFF2196F3); // For water intake

  // Custom MaterialColor for primary swatch
  static const MaterialColor primaryMaterialColor = MaterialColor(
    0xFF4CAF50,
    <int, Color>{
      50: Color(0xFFE8F5E9),
      100: Color(0xFFC8E6C9),
      200: Color(0xFFA5D6A7),
      300: Color(0xFF81C784),
      400: Color(0xFF66BB6A),
      500: Color(0xFF4CAF50),
      600: Color(0xFF43A047),
      700: Color(0xFF388E3C),
      800: Color(0xFF2E7D32),
      900: Color(0xFF1B5E20),
    },
  );
}